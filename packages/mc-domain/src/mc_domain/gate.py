"""前置闸门：先判「是否属于医疗器械」，再判管理类别。

顺序来自《医疗器械分类规则》——先看预期用途是否落入医疗器械定义
（疾病的预防、诊断、治疗、监护、缓解、补偿、替代、调节等），
落入才谈类别。跳过这一步直接把产品往目录里套，会让"日常用品"被迫归到某一类。

## 为什么这里只做高精度规则，不做全覆盖

判定"是不是医疗器械"本质上是语义问题。同一个「生理氯化钠溶液」，
官方界定为**不作为医疗器械**，而它的描述里明明有「用作雾化治疗中的吸入剂」——
关键词规则在这类样本上必然失效。

所以本模块的定位是**高精度前置拦截**：
- 只在表述明确时判 `not_device`（如"无治疗作用""不涉及疾病诊断"）
- 其余一律 `uncertain`，交给后续阶段，并在评测中如实统计门控召回率

「能承认不确定」不是缺陷。把它做成确定性的黑盒，评测时会得到虚高的指标，
上线后会在真实难例上系统性翻车。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mc_domain.enums import ClassificationOutcome
from mc_domain.schemas import Evidence, GateDecision, ProductProfile

RULE_KEY = "device_definition_gate"

#: 高精度否定表述。只收「官方界定口径里明确出现、且极少有歧义」的说法。
#: 每加一条都要能在评测集上验证它是高精度的，否则宁可留在 uncertain。
_NOT_DEVICE_MARKERS: tuple[tuple[str, str], ...] = (
    ("不作为医疗器械", "企业或官方明确表述不作为医疗器械"),
    ("不属于医疗器械", "明确表述不属于医疗器械"),
    ("非医疗用途", "明确表述非医疗用途"),
    ("不具备医疗用途", "明确表述不具备医疗用途"),
    ("不具有医疗用途", "明确表述不具有医疗用途"),
    ("无医疗用途", "明确表述无医疗用途"),
    ("不涉及疾病", "明确表述不涉及疾病诊断或治疗"),
    ("不用于疾病", "明确表述不用于疾病"),
    ("无治疗作用", "明确表述无治疗作用"),
    ("不作诊断", "明确表述不作诊断用途"),
)

#: 只看这些字段。产品名称里出现"治疗仪"不代表它就是器械，结构组成更无关。
_SCOPE_FIELDS = ("intended_use", "principle", "structure_composition")


@dataclass
class GateResult:
    decision: GateDecision
    #: 便捷出口：decision.verdict == "not_device" 时为 NOT_A_DEVICE，否则 None
    short_circuit: ClassificationOutcome | None = None
    reasons: list[str] = field(default_factory=list)


class DeviceGate:
    """确定性闸门。无副作用、可单测、不调模型。"""

    def __init__(self, markers: tuple[tuple[str, str], ...] = _NOT_DEVICE_MARKERS) -> None:
        self._markers = markers

    def evaluate(self, profile: ProductProfile) -> GateResult:
        scoped = " ".join(
            getattr(profile, f) or "" for f in _SCOPE_FIELDS
        )
        haystack = f"{scoped} {profile.raw_text}".strip()

        if not haystack:
            return GateResult(
                decision=GateDecision(
                    verdict="uncertain",
                    rule_key=RULE_KEY,
                    evidence=[
                        Evidence(
                            kind="rule",
                            ref=RULE_KEY,
                            text="没有任何可用于判断预期用途的文本",
                        )
                    ],
                ),
                reasons=["empty_input"],
            )

        for marker, rationale in self._markers:
            if marker in haystack:
                return GateResult(
                    decision=GateDecision(
                        verdict="not_device",
                        matched=marker,
                        rule_key=RULE_KEY,
                        evidence=[
                            Evidence(
                                kind="rule",
                                ref=RULE_KEY,
                                text=f"命中「{marker}」：{rationale}",
                            )
                        ],
                    ),
                    short_circuit=ClassificationOutcome.NOT_A_DEVICE,
                    reasons=[f"marker:{marker}"],
                )

        return GateResult(
            decision=GateDecision(
                verdict="uncertain",
                rule_key=RULE_KEY,
                evidence=[
                    Evidence(
                        kind="rule",
                        ref=RULE_KEY,
                        text="未命中确定性否定表述；无法仅凭规则判定是否属于医疗器械，按器械继续分类",
                    )
                ],
            ),
            reasons=["no_negative_marker"],
        )


__all__ = ["DeviceGate", "GateResult", "RULE_KEY"]
