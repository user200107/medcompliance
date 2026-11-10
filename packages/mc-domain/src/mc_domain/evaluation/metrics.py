"""评测指标。

三件事和算指标本身一样重要，都体现在这里的实现里：

1. **样本数带置信区间。** n=200 时 95% 区间约 ±7%，只报一个点值会让人高估精度。
2. **拒答单独成桶。** 系统说"信息不足，我再问你两个问题"不是错误，
   但也不能算对——把它塞进任何一边都是在操纵数字。所以报 abstain 率 + coverage。
3. **分类别指标。** 类别不平衡时 overall accuracy 会被多数类吃掉。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from mc_domain.enums import ClassificationOutcome
from mc_domain.evaluation.labels import ExpectedOutcome


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间。n 很小时比正态近似靠谱得多。"""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class ConfusionCell:
    gold: ExpectedOutcome
    pred: ClassificationOutcome
    n: int = 0


@dataclass
class GateMetrics:
    """「是不是医疗器械」的二分类。

    positive = 是医疗器械。NOT_A_DEVICE 为 negative。
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0

    def accuracy_ci(self) -> tuple[float, float]:
        return wilson_interval(self.tp + self.tn, self.n)

    def render(self) -> str:
        lo, hi = self.accuracy_ci()
        return "\n".join(
            [
                f"样本 n={self.n}",
                f"  准确率 {self.accuracy:.1%}  (95% CI {lo:.1%}–{hi:.1%})",
                f"  精确率 {self.precision:.1%}   召回率 {self.recall:.1%}   F1 {self.f1:.3f}",
                f"  特异度 {self.specificity:.1%}",
                f"  混淆：TP={self.tp}  FP={self.fp}  TN={self.tn}  FN={self.fn}",
            ]
        )


@dataclass
class ClassMetrics:
    """Ⅰ/Ⅱ/Ⅲ 三分类（Top-1 / Top-3）。"""

    n: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    per_class: dict[ClassificationOutcome, dict[str, int]] = field(default_factory=dict)

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.n if self.n else 0.0

    @property
    def top3_accuracy(self) -> float:
        return self.top3_correct / self.n if self.n else 0.0

    def top1_ci(self) -> tuple[float, float]:
        return wilson_interval(self.top1_correct, self.n)

    def class_prf(self, outcome: ClassificationOutcome) -> tuple[float, float, float]:
        c = self.per_class.get(outcome, {"tp": 0, "fp": 0, "fn": 0})
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    def macro_f1(self) -> float:
        scores = [self.class_prf(o)[2] for o in self.per_class]
        return sum(scores) / len(scores) if scores else 0.0

    def render(self) -> str:
        lo, hi = self.top1_ci()
        lines = [
            f"样本 n={self.n}",
            f"  Top-1 {self.top1_accuracy:.1%}  (95% CI {lo:.1%}–{hi:.1%})",
            f"  Top-3 {self.top3_accuracy:.1%}",
            f"  Macro-F1 {self.macro_f1():.3f}",
        ]
        if self.per_class:
            lines.append("  分类别：")
            for outcome in sorted(self.per_class, key=lambda o: o.value):
                p, r, f = self.class_prf(outcome)
                c = self.per_class[outcome]
                lines.append(
                    f"    {outcome.value:<4} 支持度 {c['tp'] + c['fn']:4d}  "
                    f"P {p:.1%}  R {r:.1%}  F1 {f:.3f}"
                )
        return "\n".join(lines)


@dataclass
class AbstainStats:
    """系统拒答（转追问）与无匹配的分布。既不是对也不是错。"""

    total: int = 0
    abstained: int = 0
    no_match: int = 0

    @property
    def coverage(self) -> float:
        """真正给出结论的比例。coverage 很低时，任何准确率都不值得看。"""
        return (self.total - self.abstained - self.no_match) / self.total if self.total else 0.0

    def render(self) -> str:
        return (
            f"  覆盖 {self.coverage:.1%}（n={self.total}）  "
            f"转追问 {self.abstained}  无匹配 {self.no_match}"
        )


def gate_metrics(
    pairs: Iterable[tuple[bool | None, bool | None]]
) -> GateMetrics:
    """(gold_is_device, pred_is_device)。任一为 None 的样本跳过。"""
    m = GateMetrics()
    for gold, pred in pairs:
        if gold is None or pred is None:
            continue
        if gold and pred:
            m.tp += 1
        elif gold and not pred:
            m.fn += 1
        elif not gold and pred:
            m.fp += 1
        else:
            m.tn += 1
    return m


def class_metrics(
    samples: Sequence[tuple[ExpectedOutcome, list[ClassificationOutcome]]],
) -> ClassMetrics:
    """samples: (gold, 按名次排列的预测候选)。gold 必须落在 class_scope 内。"""
    m = ClassMetrics(n=len(samples))
    for gold, preds in samples:
        gold_outcome = gold.system_outcome
        if gold_outcome is None:
            continue
        top1 = preds[0] if preds else None
        if top1 == gold_outcome:
            m.top1_correct += 1
        if gold_outcome in preds[:3]:
            m.top3_correct += 1

        bucket = m.per_class.setdefault(gold_outcome, {"tp": 0, "fp": 0, "fn": 0})
        if top1 == gold_outcome:
            bucket["tp"] += 1
        else:
            bucket["fn"] += 1

    # 每个类别的 fp：预测成该类但金标不是
    predicted_count: dict[ClassificationOutcome, int] = defaultdict(int)
    for _gold, preds in samples:
        if preds:
            predicted_count[preds[0]] += 1
    for outcome, bucket in m.per_class.items():
        bucket["fp"] = max(0, predicted_count.get(outcome, 0) - bucket["tp"])

    return m


def confusion_matrix(samples: Sequence[tuple[ExpectedOutcome, ClassificationOutcome | None]]) -> list[ConfusionCell]:
    cells: dict[tuple[ExpectedOutcome, ClassificationOutcome], int] = defaultdict(int)
    for gold, pred in samples:
        if pred is None:
            continue
        cells[(gold, pred)] += 1
    return [
        ConfusionCell(gold=g, pred=p, n=n)
        for (g, p), n in sorted(cells.items(), key=lambda kv: (-kv[1], kv[0][0].value))
    ]


__all__ = [
    "AbstainStats",
    "ClassMetrics",
    "ConfusionCell",
    "GateMetrics",
    "class_metrics",
    "confusion_matrix",
    "gate_metrics",
    "wilson_interval",
]
