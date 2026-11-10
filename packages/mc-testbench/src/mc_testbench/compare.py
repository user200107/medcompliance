"""运行对比与回归检测。

三个刻意的设计：

1. **方向来自声明**（`MetricSpec`），不猜。未声明的指标判 `unknown` 并警告。
2. **容差**：浮点抖动不算退化。没有容差的工具会在噪声上报警，然后被人关掉。
3. **不可比就拒绝**：task 不同、或数据指纹不同 → 退出码 **2**，而不是照样出表。
   把两次不同数据上的评测并排比较，是这类工具最危险的失败方式——
   它不会报错，只会给你一个错误的结论。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from mc_testbench.record import RunRecord
from mc_testbench.spec import MetricSpec, TaskSpec

Verdict = Literal["improved", "regressed", "unchanged", "new", "dropped", "unknown"]

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INCOMPARABLE = 2

#: 比例指标的默认容差：小于 0.1 个百分点视为持平，避免把抖动当退化
DEFAULT_TOLERANCE = 0.001


@dataclass
class Delta:
    key: str
    label: str
    unit: str
    before: float | None
    after: float | None
    diff: float | None
    verdict: Verdict

    @property
    def is_regression(self) -> bool:
        return self.verdict == "regressed"


@dataclass
class Comparison:
    base: RunRecord
    head: RunRecord
    deltas: list[Delta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    comparable: bool = True

    @property
    def regressions(self) -> list[Delta]:
        return [d for d in self.deltas if d.verdict == "regressed"]

    @property
    def improvements(self) -> list[Delta]:
        return [d for d in self.deltas if d.verdict == "improved"]

    @property
    def exit_code(self) -> int:
        if not self.comparable:
            return EXIT_INCOMPARABLE
        return EXIT_REGRESSION if self.regressions else EXIT_OK

    def summary(self) -> str:
        if not self.comparable:
            return "不可比对：" + "；".join(self.warnings or ["原因未知"])
        parts = []
        if self.regressions:
            parts.append(f"{len(self.regressions)} 项退化")
        if self.improvements:
            parts.append(f"{len(self.improvements)} 项改进")
        return "、".join(parts) if parts else "无变化"


def _display_width(text: str) -> int:
    """中文字符占两格。不算宽度就没法对齐，表格会歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    padding = max(0, width - _display_width(text))
    if align == "right":
        return " " * padding + text
    return text + " " * padding


def format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "ratio":
        return f"{value:.1%}"
    if unit == "ms":
        return f"{value:.0f}ms"
    return f"{value:g}"


def format_diff(diff: float | None, unit: str) -> str:
    if diff is None:
        return "—"
    if unit == "ratio":
        return f"{diff * 100:+.1f}pp"
    if unit == "ms":
        return f"{diff:+.0f}ms"
    return f"{diff:+g}"


_ARROW = {
    "improved": "▲",
    "regressed": "▼",
    "unchanged": "=",
    "new": "＋",
    "dropped": "－",
    "unknown": "?",
}

_VERDICT_LABEL = {
    "improved": "改进",
    "regressed": "退化",
    "unchanged": "持平",
    "new": "新增",
    "dropped": "缺失",
    "unknown": "方向未知",
}


def compare(
    base: RunRecord,
    head: RunRecord,
    spec: TaskSpec,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Comparison:
    result = Comparison(base=base, head=head)

    if base.task != head.task:
        result.comparable = False
        result.warnings.append(f"任务不同：{base.task} vs {head.task}")
        return result

    # 数据指纹不一致 → 不可比。这一步是刻意"挡住"的，不是警告了事。
    for name in sorted(set(base.datasets) | set(head.datasets)):
        before = base.datasets.get(name)
        after = head.datasets.get(name)
        if before != after:
            result.comparable = False
            result.warnings.append(f"数据集 {name} 不同：{before} vs {after}")

    keys: list[str] = list(spec.metric_keys)
    for record in (base, head):
        for key in record.metrics:
            if key not in keys:
                keys.append(key)

    for key in keys:
        metric = spec.metric(key)
        before = base.metrics.get(key)
        after = head.metrics.get(key)

        if metric is None:
            label, unit = key, "count"
        else:
            label, unit = metric.label, metric.unit

        if before is None and after is None:
            continue
        if before is None:
            result.deltas.append(Delta(key, label, unit, None, after, None, "new"))
            continue
        if after is None:
            result.deltas.append(Delta(key, label, unit, before, None, None, "dropped"))
            continue

        diff = after - before
        if metric is None:
            verdict: Verdict = "unknown"
            result.warnings.append(f"指标 {key} 未声明方向，不判断改进/退化")
        elif abs(diff) <= tolerance:
            verdict = "unchanged"
        elif metric.is_better(before, after):
            verdict = "improved"
        else:
            verdict = "regressed"

        result.deltas.append(Delta(key, label, unit, before, after, diff, verdict))

    return result


def render(comparison: Comparison) -> str:
    base, head = comparison.base, comparison.head
    lines = [
        f"基线  {base.run_id}  task={base.task}  {base.created_at}"
        f"  config={_brief(base.config)}",
        f"当前  {head.run_id}  task={head.task}  {head.created_at}"
        f"  config={_brief(head.config)}",
        "",
    ]

    header = ("指标", "基线", "当前", "变化", "判定")
    widths = [_display_width(h) for h in header]
    rows: list[tuple[str, str, str, str, str]] = []
    for delta in comparison.deltas:
        row = (
            delta.label,
            format_value(delta.before, delta.unit),
            format_value(delta.after, delta.unit),
            format_diff(delta.diff, delta.unit),
            f"{_ARROW[delta.verdict]} {_VERDICT_LABEL[delta.verdict]}",
        )
        rows.append(row)
        widths = [max(w, _display_width(cell)) for w, cell in zip(widths, row)]

    lines.append("  ".join(_pad(h, w) for h, w in zip(header, widths)).rstrip())
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(
            "  ".join(
                _pad(cell, w, "right") if idx in (1, 2, 3) else _pad(cell, w)
                for idx, (cell, w) in enumerate(zip(row, widths))
            ).rstrip()
        )

    lines.append("")
    if comparison.warnings:
        for warning in comparison.warnings:
            lines.append(f"⚠ {warning}")
        lines.append("")
    lines.append(f"→ {comparison.summary()}")
    if comparison.exit_code == EXIT_INCOMPARABLE:
        lines.append("→ 退出码 2（不可比，拒绝给出结论）")
    elif comparison.exit_code == EXIT_REGRESSION:
        lines.append("→ 退出码 1（存在退化，建议阻断合并）")
    else:
        lines.append("→ 退出码 0")
    return "\n".join(lines)


def _brief(config: dict) -> str:
    if not config:
        return "{}"
    return "{" + ", ".join(f"{k}={v}" for k, v in sorted(config.items())) + "}"
