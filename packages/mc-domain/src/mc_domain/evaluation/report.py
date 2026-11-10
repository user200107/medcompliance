"""把指标渲染成 Markdown 报告。

报告里必须同时出现三样东西，否则读者无法判断数字真不真：
1. **数据集指纹**（记录数 / SHA-256 / 时间范围）——证明指标是在哪批数据上跑的；
2. **多数类基线**——没有基线，任何准确率都无法解释；
3. **排除项说明**——哪些样本被剔除了、为什么。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mc_domain.evaluation.labels import LabelDistribution
from mc_domain.evaluation.metrics import (
    AbstainStats,
    ClassMetrics,
    ConfusionCell,
    GateMetrics,
)


@dataclass
class EvalReport:
    fingerprint: dict[str, object]
    distribution: LabelDistribution
    gate: GateMetrics
    classes: ClassMetrics
    abstain: AbstainStats
    sampled: int = 0
    catalog: dict[str, object] = field(default_factory=dict)
    confusion: list[ConfusionCell] = field(default_factory=list)
    baselines: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    # -- 渲染 ---------------------------------------------------------------

    def to_markdown(self) -> str:
        parts: list[str] = []
        parts.append("# 分类界定结果评测报告\n")
        parts.append(f"生成时间：{self.generated_at:%Y-%m-%d %H:%M}\n")

        parts.append("## 1. 数据集\n")
        fp = self.fingerprint
        parts.append(f"- 来源：{fp.get('source')}")
        parts.append(f"- 链接：{fp.get('source_url')}")
        parts.append(f"- 用途：{fp.get('usage')}")
        parts.append(f"- 全量记录：{fp.get('records')}")
        parts.append(f"- 公告批次：{fp.get('notice_batches')}")
        parts.append(f"- 时间范围：{fp.get('date_range')}")
        parts.append(f"- 数据集 SHA-256：`{fp.get('sha256')}`")
        parts.append(f"- 本次评测抽样：{self.sampled} 条\n")

        if self.catalog:
            parts.append("### 分类目录\n")
            parts.append(f"- 来源：{self.catalog.get('path')}")
            parts.append(f"- 节点数：{self.catalog.get('nodes')}")
            parts.append(f"- 可定类节点：{self.catalog.get('classifiable')}")
            parts.append(f"- 目录版本：{self.catalog.get('version')}\n")

        parts.append("## 2. 标签分布\n")
        parts.append("```")
        parts.append(self.distribution.render())
        parts.append("```\n")

        parts.append("## 3. 门控：是否属于医疗器械\n")
        parts.append("```")
        parts.append(self.gate.render())
        parts.append("```\n")
        if self.baselines:
            parts.append("基线对照：\n")
            for name, value in self.baselines.items():
                parts.append(f"- {name}：{value:.1%}")
            parts.append("")
            parts.append(
                "> 只看准确率会被多数类吃掉。上面的精确率/召回率才是门控的真实水平。\n"
            )

        parts.append("## 4. 管理类别：Ⅰ / Ⅱ / Ⅲ\n")
        parts.append("```")
        parts.append(self.classes.render())
        parts.append("```\n")
        parts.append("```")
        parts.append(self.abstain.render())
        parts.append("```\n")

        if self.confusion:
            parts.append("混淆对（金标 → 预测，取前 10）：\n")
            parts.append("| 金标 | 预测 | 条数 |")
            parts.append("|---|---|---|")
            for cell in self.confusion[:10]:
                parts.append(f"| {cell.gold.value} | {cell.pred.value} | {cell.n} |")
            parts.append("")

        if self.notes:
            parts.append("## 5. 结论与限制\n")
            for note in self.notes:
                parts.append(f"- {note}")
            parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "fingerprint": self.fingerprint,
            "catalog": self.catalog,
            "sampled": self.sampled,
            "distribution": {
                o.value: n for o, n in self.distribution.counts.items()
            },
            "gate": {
                "n": self.gate.n,
                "accuracy": self.gate.accuracy,
                "precision": self.gate.precision,
                "recall": self.gate.recall,
                "f1": self.gate.f1,
                "specificity": self.gate.specificity,
                "tp": self.gate.tp,
                "fp": self.gate.fp,
                "tn": self.gate.tn,
                "fn": self.gate.fn,
            },
            "classes": {
                "n": self.classes.n,
                "top1": self.classes.top1_accuracy,
                "top3": self.classes.top3_accuracy,
                "macro_f1": self.classes.macro_f1(),
            },
            "abstain": {
                "total": self.abstain.total,
                "abstained": self.abstain.abstained,
                "no_match": self.abstain.no_match,
                "coverage": self.abstain.coverage,
            },
            "baselines": self.baselines,
            "notes": self.notes,
        }


__all__ = ["EvalReport"]
