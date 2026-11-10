"""标签归一化：把官方原文的「产品类别」映射成可比较的期望结论。

## 为什么数据里是 5 种、系统只能产出 4 种

实测官方《分类界定结果汇总》里出现过的原文标签：
    `Ⅰ` `Ⅱ` `Ⅲ` `不作为医疗器械` `药械组合`

其中 `药械组合`（药械组合产品）是**独立的监管结论**，不等于任何管理类别，
系统目前不产出它。这不是靠改名能糊过去的差异，必须显式统计：

    标签空间 5 类  ┃  系统输出空间 4 类（Ⅰ/Ⅱ/Ⅲ/不作为医疗器械）

所以这里用 `ExpectedOutcome`（标签空间）与 `ClassificationOutcome`（输出空间）
两个枚举分开表达，`unsupported` 类别的样本从指标中**排除并计数**，
而不是硬塞进某一类去凑准确率。

## 两条原则

1. 认不出的标签必须落到 `OTHER` 并计入 `unmapped`，**绝不猜**。
   猜错标签比少几条样本危险得多——它会让"准确率"整体失真。
2. 归一化规则必须有单测覆盖（tests/test_labels.py）。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from mc_domain.enums import ClassificationOutcome

#: 全角罗马数字 → 半角
_ROMAN_MAP = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V"})
_WS_RE = re.compile(r"[\s\u3000]+")

_NOT_DEVICE_MARKERS = (
    "不作为医疗器械",
    "不属于医疗器械",
    "不作为器械",
    "非医疗器械",
    "不按医疗器械",
    "不按照医疗器械",   # 「建议不按照医疗器械管理」
    "不按照器械",
    "不具有医疗用途",
    "不具备医疗用途",
    "无医疗用途",
)

_COMBINATION_MARKERS = ("药械组合", "药械结合")

#: 「不单独作为医疗器械」：通常是部件/附件，随主机一并管理。
#: 它既不是"不是器械"，也不能直接给类别——必须单列，否则会污染两类指标。
#: 「建议不单独申报」是同一族表述（企业在申报策略上被建议与主机合并）。
_NOT_ALONE_MARKERS = (
    "不单独作为医疗器械",
    "不单独按医疗器械",
    "不单独作为器械",
    "建议不单独申报",
    "不单独申报",
)

#: 「视具体情况而定」：**官方自己不给唯一结论**。
#: 这类样本必须从准确率中剔除——要求系统比监管机构更确定是不公平的评测。
_DEPENDS_MARKERS = ("视具体情况而定", "根据具体情况", "具体情况确定")


class ExpectedOutcome(str, Enum):
    """**标签空间**。比系统输出空间多若干格——这个不对称本身就是要被量化的东西。"""

    CLASS_I = "I"
    CLASS_II = "II"
    CLASS_III = "III"
    NOT_A_DEVICE = "not_a_device"
    NOT_ALONE = "not_alone"
    COMBINATION = "combination"
    DEPENDS = "depends"
    OTHER = "other"

    @property
    def system_outcome(self) -> ClassificationOutcome | None:
        """能映射到系统输出空间就返回；其余返回 None。"""
        if self in _UNSUPPORTED:
            return None
        return ClassificationOutcome(self.value)

    @property
    def in_gate_scope(self) -> bool:
        """能否参与「是不是医疗器械」这项判定。"""
        return self not in _UNSUPPORTED

    @property
    def in_class_scope(self) -> bool:
        """能否参与「Ⅰ/Ⅱ/Ⅲ」类别判定。"""
        return self in (ExpectedOutcome.CLASS_I, ExpectedOutcome.CLASS_II, ExpectedOutcome.CLASS_III)

    @property
    def is_device(self) -> bool | None:
        """真值意义上的「是不是医疗器械」。无法二分的返回 None。"""
        if not self.in_gate_scope:
            return None
        return self is not ExpectedOutcome.NOT_A_DEVICE


def normalize_label(raw: str) -> ExpectedOutcome:
    """无法识别的返回 `OTHER`（调用方需计入 unmapped，禁止静默丢弃）。"""
    if not raw or not raw.strip():
        return ExpectedOutcome.OTHER

    text = _WS_RE.sub("", raw).translate(_ROMAN_MAP)

    # 顺序有讲究：越具体的表述越先判。
    # 「不单独作为医疗器械」里含「作为医疗器械」，若不先判会被误当普通非器械。
    if any(m in text for m in _DEPENDS_MARKERS):
        return ExpectedOutcome.DEPENDS
    if any(m in text for m in _NOT_ALONE_MARKERS):
        return ExpectedOutcome.NOT_ALONE
    if any(m in text for m in _COMBINATION_MARKERS):
        return ExpectedOutcome.COMBINATION
    if any(m in text for m in _NOT_DEVICE_MARKERS):
        return ExpectedOutcome.NOT_A_DEVICE

    # 顺序敏感：III 必须在 II、I 之前判，否则 "III" 会先命中 "II"
    if "III" in text or "三" in text or "3" in text:
        return ExpectedOutcome.CLASS_III
    if "II" in text or "二" in text or "2" in text:
        return ExpectedOutcome.CLASS_II
    if "I" in text or "一" in text or "1" in text:
        return ExpectedOutcome.CLASS_I

    return ExpectedOutcome.OTHER


#: 系统输出空间不包含的结论。定义在类之后，因为它引用枚举成员本身。
_UNSUPPORTED = frozenset(
    {
        ExpectedOutcome.NOT_ALONE,
        ExpectedOutcome.COMBINATION,
        ExpectedOutcome.DEPENDS,
        ExpectedOutcome.OTHER,
    }
)


@dataclass
class LabelDistribution:
    """先测标签分布再谈准确率——不平衡时 overall accuracy 毫无意义。"""

    total: int = 0
    counts: Counter[ExpectedOutcome] = field(default_factory=Counter)
    unmapped: Counter[str] = field(default_factory=Counter)

    @property
    def mapped(self) -> int:
        return sum(self.counts.values()) - self.counts[ExpectedOutcome.OTHER]

    @property
    def gate_scope(self) -> int:
        return sum(n for o, n in self.counts.items() if o.in_gate_scope)

    @property
    def class_scope(self) -> int:
        return sum(n for o, n in self.counts.items() if o.in_class_scope)

    @property
    def unsupported(self) -> int:
        """系统无法表达的结论数。必须单独报，不能被摊进任何一类的分母。"""
        return sum(n for o, n in self.counts.items() if o in _UNSUPPORTED)

    def share(self, outcome: ExpectedOutcome) -> float:
        return self.counts[outcome] / self.total if self.total else 0.0

    @property
    def device_share(self) -> float:
        """器械 vs 非器械的比例，决定单一阈值/单类基线是不是骗人。"""
        if not self.gate_scope:
            return 0.0
        device = sum(
            n for o, n in self.counts.items() if o.in_class_scope
        )
        return device / self.gate_scope

    def is_imbalanced(self, threshold: float = 0.8) -> bool:
        if not self.gate_scope:
            return False
        return max(self.device_share, 1 - self.device_share) > threshold

    def render(self) -> str:
        lines = [
            f"总记录 {self.total}",
            f"  标签空间可映射 {self.mapped}  门控范围 {self.gate_scope}  类别范围 {self.class_scope}  系统不支持 {self.unsupported}",
            "",
        ]
        for outcome, n in self.counts.most_common():
            flag = "" if outcome.in_gate_scope else "  ← 系统不支持，已排除出指标"
            lines.append(f"  {outcome.value:<14} {n:6d}  {self.share(outcome):6.1%}{flag}")
        if self.gate_scope:
            lines.append(f"\n  器械占比 {self.device_share:.1%}（门控范围内）")
            if self.is_imbalanced():
                lines.append("  ⚠ 类别严重不平衡：只报 total 准确率会失真，必须分类别报")
        if self.unmapped:
            lines.append("\n  未识别标签（需补归一化规则）：")
            for raw, n in self.unmapped.most_common(15):
                lines.append(f"    {raw!r:<28} {n}")
        return "\n".join(lines)


def build_distribution(raw_labels: list[str]) -> LabelDistribution:
    dist = LabelDistribution(total=len(raw_labels))
    for raw in raw_labels:
        outcome = normalize_label(raw)
        dist.counts[outcome] += 1
        if outcome is ExpectedOutcome.OTHER:
            dist.unmapped[(raw or "(空)").strip()] += 1
    return dist


__all__ = [
    "ExpectedOutcome",
    "LabelDistribution",
    "build_distribution",
    "normalize_label",
]
