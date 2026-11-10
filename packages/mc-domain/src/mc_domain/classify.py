"""P1 分类判定流水线。

核心主张：**模型只做翻译，判定交给确定性引擎。**

    normalize   模型把用户口语翻译成结构化产品画像（唯一允许调模型的地方）
    coarse      检索粗筛子目录                ┐
    exclude     排除说明命中即剔除              │ 全部确定性、可复现、可单测
    match       在存活子树内匹配到叶子类别       │
    rule        分类规则裁决（取最高 / 取最具体） ┘
    rank        输出 Top-K + 依据链

每个阶段都产出 StepTrace；被淘汰的候选进 rejected 并带理由。
「为什么不是它」和「为什么是它」一样是交付物。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from mc_domain.enums import ClassificationOutcome, ManagementClass
from mc_domain.decision import (
    PRESETS,
    RiskPreference,
    get_preference,
    select_candidate,
)
from mc_domain.gate import DeviceGate
from mc_domain.schemas import (
    ApplicableRegulation,
    Candidate,
    ClassificationResult,
    Evidence,
    MissingSlot,
    ProductProfile,
    RejectedCandidate,
    StepTrace,
)
from mc_domain.tree import ClassificationTree, CodeNode, ExclusionMatcher

ENGINE_ID = "tree_rule_v1"
TOP_K = 3
#: Top-1 与 Top-2 分数差小于该比例即视为「分不开」，转人工/追问。
AMBIGUITY_MARGIN = 0.15


# ---------------------------------------------------------------------------
# 依赖协议：真实实现走 Milvus+BM25 与 LLM，测试里换成假实现
# ---------------------------------------------------------------------------


class CodeRetriever(Protocol):
    def search_codes(
        self, query: str, *, limit: int, allowed_codes: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """返回 [(code, score)]，score 越大越相关。allowed_codes 为 None 表示全库。"""
        ...


class ProfileNormalizer(Protocol):
    async def normalize(self, raw_text: str) -> ProductProfile:
        """把自由文本规范化为产品画像。失败必须抛异常，**不许返回空画像冒充成功**。"""
        ...


ApplicabilityLoader = Callable[[str, date], Awaitable[list[ApplicableRegulation]]]


# ---------------------------------------------------------------------------
# 规则即数据：可引用、可审计、可单测
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    key: str
    title: str
    basis: str

    def as_evidence(self) -> Evidence:
        return Evidence(kind="rule", ref=self.key, text=f"{self.title}（依据：{self.basis}）")


R_HIGHEST_CLASS = Rule(
    key="highest_class",
    title="同一产品适用多个管理类别时，按较高类别管理",
    basis="《医疗器械分类规则》关于多类别适用的规定",  # 正式引用时填入准确条款号
)
R_MOST_SPECIFIC = Rule(
    key="most_specific",
    title="候选存在祖孙关系时，取更具体的类别",
    basis="分类目录的层级语义：下层类别的描述更精确",
)
R_COMBINATION = Rule(
    key="combination",
    title="组合产品按主要功能确定类别；主要功能无法确定时按较高类别",
    basis="《医疗器械分类规则》关于组合产品的规定",
)
R_EXCLUSION = Rule(
    key="exclusion",
    title="命中类别排除说明的产品不属于该类别",
    basis="分类目录各级「本类不包括」说明",
)


# ---------------------------------------------------------------------------
# 缺槽位时的追问设计
# ---------------------------------------------------------------------------

_SLOT_QUESTIONS: dict[str, MissingSlot] = {
    "intended_use": MissingSlot(
        slot="intended_use",
        question="这个产品的预期用途是什么？（用于什么部位、达到什么目的）",
        example="例：用于眼科手术中观察眼底",
    ),
    "contact_type": MissingSlot(
        slot="contact_type",
        question="使用时会接触人体吗？接触到什么程度？",
        options=["不接触人体", "表面接触", "间接接触（经管路/体液）", "侵入人体", "植入体内"],
    ),
    "energy_type": MissingSlot(
        slot="energy_type",
        question="产品本身是否带电/产生能量？",
        options=["有源（需要电源驱动）", "无源（不依赖电源）"],
    ),
    "contact_duration": MissingSlot(
        slot="contact_duration",
        question="单次接触人体的时间大致多久？",
        options=["60 分钟以内", "60 分钟以上、30 天以内", "30 天以上"],
    ),
    "is_sterile": MissingSlot(
        slot="is_sterile",
        question="产品是无菌提供的吗？",
        options=["无菌", "非无菌"],
    ),
}


_SLOT_LABELS: dict[str, str] = {
    "intended_use": "预期用途",
    "contact_type": "接触类型",
    "contact_duration": "接触时长",
    "energy_type": "有源/无源",
    "is_sterile": "是否无菌",
    "is_reusable": "是否可重复使用",
}


def _missing_slots(profile: ProductProfile) -> list[MissingSlot]:
    """阻断性缺失的问题。每轮最多问 2 个——问多了用户会放弃。"""
    return [_SLOT_QUESTIONS[s] for s in profile.missing_slots() if s in _SLOT_QUESTIONS][:2]


def _advisory_questions(profile: ProductProfile) -> list[MissingSlot]:
    """标注性缺失的问题。即使已经出了结论，也提供"想更准就补这两项"。"""
    return [_SLOT_QUESTIONS[s] for s in profile.advisory_missing() if s in _SLOT_QUESTIONS][:2]


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------


class ClassificationPipeline:
    def __init__(
        self,
        tree: ClassificationTree,
        retriever: CodeRetriever,
        normalizer: ProfileNormalizer,
        applicability_loader: ApplicabilityLoader | None = None,
        *,
        exclusion_matcher: ExclusionMatcher | None = None,
        gate: DeviceGate | None = None,
        preference: RiskPreference | None = None,
    ) -> None:
        self.tree = tree
        self.retriever = retriever
        self.normalizer = normalizer
        self.applicability_loader = applicability_loader
        self.exclusions = exclusion_matcher or ExclusionMatcher()
        self.gate = gate or DeviceGate()
        self.preference = preference or get_preference(None)

    async def classify(
        self,
        raw_text: str,
        *,
        as_of: date | None = None,
        profile: ProductProfile | None = None,
        coarse_limit: int = 6,
        leaf_limit: int = 10,
        risk: str | None = None,
    ) -> ClassificationResult:
        started = time.monotonic()
        as_of = as_of or date.today()
        steps: list[StepTrace] = []

        preferred = get_preference(risk) if risk else self.preference
        preferred_label = next((n for n, p in PRESETS.items() if p == preferred), "custom")

        # --- 0. normalize ------------------------------------------------
        if profile is None:
            profile = await self.normalizer.normalize(raw_text)
        if profile.raw_text == "":
            profile.raw_text = raw_text

        steps.append(
            StepTrace(
                stage="normalize",
                summary=f"识别到 {len(profile.missing_slots())} 个必需槽位缺失",
            )
        )

        # --- 0.5 gate：先判「是否属于医疗器械」 -----------------------------
        # 顺序与《医疗器械分类规则》一致，而且**必须早于槽位检查**：
        # 判"是不是器械"只需要预期用途，不需要接触类型/有无源。
        # 先要求槽位再判闸门，会让"没有医疗用途"的产品被无意义地反复追问。
        gate_result = self.gate.evaluate(profile)
        steps.append(
            StepTrace(
                stage="gate",
                summary=f"闸门结论：{gate_result.decision.verdict}"
                + (f"（命中「{gate_result.decision.matched}」）" if gate_result.decision.matched else ""),
                rule_key=gate_result.decision.rule_key,
            )
        )
        if gate_result.short_circuit is not None:
            return ClassificationResult(
                status="classified",
                as_of=as_of,
                engine=ENGINE_ID,
                outcome=gate_result.short_circuit,
                risk_preference=preferred_label,
                gate=gate_result.decision,
                steps=steps,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )

        # 判类别才需要槽位
        if not profile.is_classifiable():
            return ClassificationResult(
                status="needs_more_info",
                as_of=as_of,
                engine=ENGINE_ID,
                risk_preference=preferred_label,
                gate=gate_result.decision,
                steps=steps,
                missing_slots=_missing_slots(profile),
            )

        query = profile.search_text()

        # --- 1. coarse：粗筛子目录 ---------------------------------------
        root_codes = {n.code for n in self.tree.roots()}
        root_hits = self.retriever.search_codes(query, limit=coarse_limit, allowed_codes=root_codes)
        if not root_hits:
            steps.append(StepTrace(stage="coarse", summary="没有任何子目录命中", kept=0))
            return ClassificationResult(
                status="no_match",
                as_of=as_of,
                engine=ENGINE_ID,
                risk_preference=preferred_label,
                steps=steps,
            )

        surviving_roots = [self.tree.get(code) for code, _ in root_hits if code in self.tree]
        steps.append(
            StepTrace(
                stage="coarse",
                summary=f"命中 {len(root_hits)} 个子目录：" + "、".join(r.name for r in surviving_roots),
                kept=len(surviving_roots),
            )
        )

        # --- 2. exclude：排除法 ------------------------------------------
        rejected: list[RejectedCandidate] = []
        kept_roots: list[CodeNode] = []
        for root in surviving_roots:
            hit = self.exclusions.match(root, query)
            if hit:
                rejected.append(
                    RejectedCandidate(
                        code=root.code,
                        name=root.name,
                        reason=f"命中排除说明：{hit.matched_term}",
                        evidence=Evidence(
                            kind="catalog_exclusion",
                            ref=root.code,
                            text=hit.as_evidence_text(),
                        ),
                    )
                )
            else:
                kept_roots.append(root)

        steps.append(
            StepTrace(
                stage="exclude",
                summary=f"排除说明剔除 {len(rejected)} 个子目录",
                kept=len(kept_roots),
                dropped=len(rejected),
                rule_key=R_EXCLUSION.key,
            )
        )
        if not kept_roots:
            return ClassificationResult(
                status="no_match",
                as_of=as_of,
                engine=ENGINE_ID,
                risk_preference=preferred_label,
                steps=steps,
                rejected=rejected,
            )

        # --- 3. match：在存活子树内找叶子 --------------------------------
        allowed: set[str] = set()
        for root in kept_roots:
            allowed.update(n.code for n in self.tree.subtree(root.code))

        leaf_hits = self.retriever.search_codes(query, limit=leaf_limit, allowed_codes=allowed)
        candidates: list[Candidate] = []
        for code, score in leaf_hits:
            node = self.tree.maybe_get(code)
            # 只有带管理类别的节点才能给出结论；纯分类容器节点不参与排名
            if node is None or node.management_class is None:
                continue
            candidates.append(
                Candidate(
                    code=node.code,
                    name=node.name,
                    name_path=node.name_path,
                    management_class=node.management_class,
                    score=round(score, 4),
                    evidence=_match_evidence(node),
                )
            )

        steps.append(
            StepTrace(
                stage="match",
                summary=f"在 {len(kept_roots)} 个子目录内匹配到 {len(candidates)} 个可定类类别",
                kept=len(candidates),
            )
        )
        if not candidates:
            return ClassificationResult(
                status="no_match",
                as_of=as_of,
                engine=ENGINE_ID,
                risk_preference=preferred_label,
                steps=steps,
                rejected=rejected,
            )

        # --- 4. rule：取最具体 → 按风险偏好取舍 --------------------------
        candidates, dropped_specific = _apply_most_specific(candidates)
        if dropped_specific:
            steps.append(
                StepTrace(
                    stage="rule",
                    summary="祖孙候选并存，保留更具体的类别",
                    kept=len(candidates),
                    dropped=dropped_specific,
                    rule_key=R_MOST_SPECIFIC.key,
                )
            )

        # --- 5. rank + 风险取舍 + 歧义判定 --------------------------------
        candidates.sort(key=lambda c: c.score, reverse=True)
        chosen, reason = select_candidate(candidates, preferred)
        steps.append(
            StepTrace(
                stage="rule",
                summary=reason,
                kept=1,
                dropped=max(0, len(candidates) - 1),
                rule_key=R_HIGHEST_CLASS.key if preferred.alpha > 0 else None,
            )
        )

        top_window = candidates[:TOP_K]
        if chosen is not None and chosen not in top_window:
            # 就高策略可能把窗口外的更高类别捞上来，必须让它出现在候选列表里
            top_window.append(chosen)
            top_window.sort(key=lambda c: c.score, reverse=True)
        candidates = top_window
        steps.append(StepTrace(stage="rank", summary=f"输出 Top-{len(candidates)}"))

        best = chosen
        if best is None:
            return ClassificationResult(
                status="no_match",
                as_of=as_of,
                engine=ENGINE_ID,
                risk_preference=preferred_label,
                gate=gate_result.decision,
                steps=steps,
                rejected=rejected,
            )

        result = ClassificationResult(
            status="classified",
            as_of=as_of,
            engine=ENGINE_ID,
            outcome=ClassificationOutcome.from_management_class(best.management_class),
            risk_preference=preferred_label,
            best=best,
            candidates=candidates,
            rejected=rejected,
            gate=gate_result.decision,
            steps=steps,
        )

        # 标注性缺失：仍然给结论，但必须把"哪些维度没确认"一起交付。
        # 悄悄吞掉等于让用户以为结论建立在完整信息上。
        advisory = profile.advisory_missing()
        if advisory:
            result.unconfirmed = advisory
            result.missing_slots = _advisory_questions(profile)
            result.caveats.append(
                "未确认的判定维度："
                + "、".join(_SLOT_LABELS.get(s, s) for s in advisory)
                + "。结论基于已提供的文本；补全后可能落到相邻类别，建议复判。"
            )

        if _is_ambiguous(candidates):
            result.status = "needs_more_info"
            result.missing_slots = [
                MissingSlot(
                    slot="intended_use",
                    question=(
                        f"「{candidates[0].name}」与「{candidates[1].name}」都可能适用，"
                        "能否补充一句预期用途（用于什么部位、达到什么目的）？"
                    ),
                )
            ]
            result.caveats.append("Top-2 候选分数接近，当前结论不足以定类。")

        if self.applicability_loader:
            result.applicable_regulations = await self.applicability_loader(
                best.code, as_of
            )

        result.elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        return result


# ---------------------------------------------------------------------------
# 规则实现（纯函数，可独立单测）
# ---------------------------------------------------------------------------


def _apply_most_specific(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """A 是 B 的祖先时丢掉 A。

    编码是层级化的（07 → 07-01 → 07-01-01），所以前缀判断就够，
    不需要再回树里查 parent。编码分隔符一旦变化，这里要同步改。
    """
    kept: list[Candidate] = []
    dropped = 0
    for cand in candidates:
        if any(
            other is not cand and other.code.startswith(cand.code + "-")
            for other in candidates
        ):
            dropped += 1
            continue
        kept.append(cand)
    return kept or candidates, dropped


def _is_ambiguous(candidates: Sequence[Candidate]) -> bool:
    if len(candidates) < 2:
        return False
    top, second = candidates[0], candidates[1]
    if top.score <= 0:
        return False
    # **只在结论层有分歧时才算歧义。**
    # 两个候选如果管理类别相同，编码再接近也不影响注册路径——
    # 而注册专员要的是路径（一类备案 / 二类省级 / 三类国家局），不是编码本身。
    # 之前的实现只看分数差，把大量"同类别、不同编码"的样本错误地变成了追问：
    # 实测 53 条里 30 条 needs_more_info，绝大多数属于这种情况。
    if top.management_class == second.management_class:
        return False
    return (top.score - second.score) / top.score < AMBIGUITY_MARGIN


def _match_evidence(node: CodeNode) -> list[Evidence]:
    ev: list[Evidence] = [
        Evidence(kind="catalog_entry", ref=node.code, text=f"{node.name_path}（{node.code}）")
    ]
    if node.examples:
        ev.append(
            Evidence(
                kind="catalog_example",
                ref=node.code,
                text="品名举例：" + "、".join(node.examples[:8]),
            )
        )
    return ev
