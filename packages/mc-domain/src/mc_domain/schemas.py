"""P1 的输入 / 输出契约，以及实体的读模型。

约定：**写模型用 ORM，读模型用 Pydantic**。跨应用传输一律走这里的 schema，
不要让 P2/P3 直接 import 另一方的 ORM 类。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mc_domain.enums import (
    CertStatus,
    CertType,
    ClassificationOutcome,
    ClauseKind,
    ContactDuration,
    ContactType,
    EnergyType,
    ManagementClass,
    ProductStatus,
    RegulationKind,
    RequirementKey,
    VersionStatus,
)


# ---------------------------------------------------------------------------
# P1 输入：产品画像
# ---------------------------------------------------------------------------

#: **阻断性**槽位。判据不是"某个字段为空"，而是"还能不能拼出检索文本"：
#: 没有可用文本就没有可匹配的东西，只能追问。
BLOCKING_SLOTS: tuple[str, ...] = ("intended_use",)

#: **标注性**槽位。它们是《医疗器械分类规则》里的判定维度，但**不参与检索**——
#: `search_text()` 只拼预期用途/名称/原理/结构/材料。
#: 所以它们缺失时应当降级出结论并明确标注，而不是把用户拦在门外。
ADVISORY_SLOTS: tuple[str, ...] = (
    "contact_type",
    "energy_type",
    "contact_duration",
    "is_sterile",
    "is_reusable",
)


class ProductProfile(BaseModel):
    """结构化后的产品描述。

    这是 P1 第 0 阶段的产物，也是 P2 / P3 复用产品信息的标准形态。
    所有槽位都允许 None —— None 表示「用户没说」，与 False / "无" 严格区分。
    """

    name: str | None = None
    intended_use: str | None = None
    principle: str | None = None
    structure_composition: str | None = None
    materials: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)

    contact_type: ContactType | None = None
    contact_duration: ContactDuration | None = None
    energy_type: EnergyType | None = None
    is_sterile: bool | None = None
    is_reusable: bool | None = None

    raw_text: str = ""

    def missing_slots(self) -> list[str]:
        """**阻断性**缺失。判据是"还能不能拼出检索文本"，而不是某个字段是否为空。

        这样定义是为了避免一种尴尬：产品名称明明能给检索用，
        却因为"接触类型未知"把用户挡在门外，问一个和当前问题无关的问题。
        """
        if self.search_text():
            return []
        return list(BLOCKING_SLOTS)

    def advisory_missing(self) -> list[str]:
        """**标注性**缺失：不影响出结论，但结论的确定性受它们限制。"""
        return [s for s in ADVISORY_SLOTS if getattr(self, s, None) is None]

    def is_classifiable(self) -> bool:
        return not self.missing_slots()

    def search_text(self) -> str:
        """给检索用的拼接文本。刻意把预期用途放在最前，它在分类里权重最高。"""
        parts = [
            self.intended_use or "",
            self.name or "",
            self.principle or "",
            self.structure_composition or "",
            " ".join(self.materials),
        ]
        return " ".join(p for p in parts if p).strip()


class MissingSlot(BaseModel):
    """缺字段时给用户看的问题。数量要少（每轮 ≤ 2 个），否则用户会放弃。"""

    slot: str
    question: str
    options: list[str] = Field(default_factory=list)
    example: str | None = None


# ---------------------------------------------------------------------------
# P1 输出：带依据链的分类结论
# ---------------------------------------------------------------------------

EvidenceKind = Literal["catalog_entry", "catalog_exclusion", "catalog_example", "rule", "regulation"]


class Evidence(BaseModel):
    """一条「凭什么这么判」的记录。P1 的价值 90% 在这里。"""

    kind: EvidenceKind
    ref: str
    """目录编码 / 规则 key / 条款号 / 文号。"""

    text: str
    """原文或规则描述。报告里直接展示，不要二次改写。"""

    source_url: str | None = None


class Candidate(BaseModel):
    code: str
    name: str
    name_path: str
    management_class: ManagementClass | None = None
    score: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)


class RejectedCandidate(BaseModel):
    """被排除的候选也要留痕——「为什么不是它」和「为什么是它」一样重要。"""

    code: str
    name: str
    reason: str
    evidence: Evidence


class StepTrace(BaseModel):
    """一个阶段的可审计记录。整条 steps 串起来就是依据链。"""

    stage: Literal["normalize", "gate", "coarse", "exclude", "match", "rule", "rank"]
    summary: str
    kept: int = 0
    dropped: int = 0
    rule_key: str | None = None


GateVerdict = Literal["device", "not_device", "uncertain"]


class GateDecision(BaseModel):
    """前置闸门：先判「是否属于医疗器械」，再判类别。

    `uncertain` 是常态而非缺陷——判定是否属于医疗器械本质上依赖预期用途的语义，
    规则只能拦下表述明确的少部分。**能承认不确定，比强行二分更有价值**，
    而且是可被评测量化的（见 evaluation/ 里的门控指标）。
    """

    verdict: GateVerdict
    matched: str | None = None
    """命中的规则词；uncertain 时为 None。"""

    evidence: list[Evidence] = Field(default_factory=list)
    rule_key: str | None = None


class ClassificationResult(BaseModel):
    status: Literal["classified", "needs_more_info", "no_match"]
    as_of: date
    engine: str = "tree_rule_v1"

    outcome: ClassificationOutcome | None = None
    """系统给出的最终结论。`NOT_A_DEVICE` 时 `best` 为 None——它不是类别，是对类别的否定。"""

    risk_preference: str = "balanced"
    """本次采用的风险偏好。写进结果是为了可审计——同一个问题换策略可能得到不同类别。"""

    unconfirmed: list[str] = Field(default_factory=list)
    """**未确认的判定维度**（接触类型 / 有源无源等）。

    非空不代表结论不可用，而是说：这些维度缺省时系统仍然给结论，
    但结论可能在相邻类别间摇摆。必须随结果一起展示，不能悄悄吞掉。
    """

    best: Candidate | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)

    gate: GateDecision | None = None
    steps: list[StepTrace] = Field(default_factory=list)
    missing_slots: list[MissingSlot] = Field(default_factory=list)
    elapsed_ms: float = 0.0

    #: 适用的法规版本（已按 as_of 过滤）。下游 P2/P3 直接用。
    applicable_regulations: list[ApplicableRegulation] = Field(default_factory=list)

    #: 必须原样展示给用户的自检提示。
    caveats: list[str] = Field(
        default_factory=lambda: [
            "分类结论依据公开分类目录与分类规则推导，仅供参考。",
            "最终管理类别以药品监督管理部门的分类界定结果为准。",
        ]
    )

    def top_k(self, k: int = 3) -> list[Candidate]:
        return self.candidates[:k]


# ---------------------------------------------------------------------------
# 实体读模型
# ---------------------------------------------------------------------------


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ClassificationCodeRead(_ORM):
    id: int
    catalog_version: str
    code: str
    level: int
    name: str
    name_path: str
    management_class: ManagementClass | None
    exclusion_note: str | None
    examples: list[str]


class ProductRead(_ORM):
    id: int
    name: str
    models: list[str]
    intended_use: str | None
    contact_type: ContactType | None
    energy_type: EnergyType | None
    classification_code_id: int | None
    management_class: ManagementClass | None
    status: ProductStatus
    created_at: datetime


class RegistrationCertRead(_ORM):
    id: int
    cert_no: str
    cert_type: CertType
    registrant_name: str
    models: list[str]
    valid_from: date | None
    valid_to: date | None
    status: CertStatus


class RegulationRead(_ORM):
    id: int
    kind: RegulationKind
    doc_no: str
    title: str
    issuer: str | None
    meta: dict


class RegulationVersionRead(_ORM):
    id: int
    regulation_id: int
    version_label: str
    publish_date: date | None
    effective_date: date | None
    expiry_date: date | None
    status: VersionStatus
    superseded_by_version_id: int | None


class ClauseRead(_ORM):
    id: int
    regulation_version_id: int
    clause_no: str
    heading: str | None
    text: str
    kind: ClauseKind
    requirement_key: RequirementKey | None


class ApplicableRegulation(BaseModel):
    """P1 → P2/P3 的交接物：某类别在 as_of 当天适用哪些法规，精确到版本与依据条款。"""

    regulation_id: int
    version_id: int
    kind: RegulationKind
    doc_no: str
    title: str
    version_label: str
    effective_date: date | None
    is_mandatory: bool
    basis_clause_no: str | None = None


ClassificationResult.model_rebuild()
