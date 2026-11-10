"""评测子包：把「准确率」变成可复现、可追问的数字。

    labels.py    标签空间（5 类）与归一化
    cases.py     从官方界定结果构造 (输入, 标签)；分层抽样；数据指纹
    metrics.py   门控 / 类别 / 拒答三类指标，带 Wilson 置信区间
    report.py    Markdown 报告

设计上刻意把「标签空间」和「系统输出空间」分开：
官方数据有 5 种结论，系统目前只能产出 4 种。强行对齐会让指标失真，
显式统计 unsupported 才能如实说明覆盖率。
"""

from __future__ import annotations

from mc_domain.evaluation.cases import (
    AdjudicationCase,
    distribution,
    fingerprint,
    load_cases,
    stratified_sample,
)
from mc_domain.evaluation.labels import (
    ExpectedOutcome,
    LabelDistribution,
    build_distribution,
    normalize_label,
)
from mc_domain.evaluation.metrics import (
    AbstainStats,
    ClassMetrics,
    ConfusionCell,
    GateMetrics,
    class_metrics,
    confusion_matrix,
    gate_metrics,
    wilson_interval,
)

__all__ = [
    "AbstainStats",
    "AdjudicationCase",
    "ClassMetrics",
    "ConfusionCell",
    "ExpectedOutcome",
    "GateMetrics",
    "LabelDistribution",
    "build_distribution",
    "class_metrics",
    "confusion_matrix",
    "distribution",
    "fingerprint",
    "gate_metrics",
    "load_cases",
    "normalize_label",
    "stratified_sample",
    "wilson_interval",
]
