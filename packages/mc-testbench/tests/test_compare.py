"""回归检测的单测。

最该钉住的两条：
1. **方向**：`undeterminable_rate` 变大是退化，不是改进；
2. **不可比就拒绝**：数据指纹不同时必须返回退出码 2，而不是给一个"看起来正常"的表。
"""

from __future__ import annotations

import pytest

from mc_testbench.compare import (
    EXIT_INCOMPARABLE,
    EXIT_OK,
    EXIT_REGRESSION,
    compare,
    format_diff,
    format_value,
    render,
)
from mc_testbench.record import RunRecord
from mc_testbench.spec import MetricSpec, TaskSpec

SPEC = TaskSpec(
    name="toy",
    title="玩具任务",
    metrics=(
        MetricSpec("recall", "召回率"),
        MetricSpec("undeterminable_rate", "不可判定占比", higher_is_better=False),
        MetricSpec("p95", "P95 延迟", unit="ms", higher_is_better=False),
    ),
    datasets=("toy_set",),
)


def rec(run_id: str, metrics: dict[str, float], *, task="toy", datasets=None) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        task=task,
        created_at="2026-11-10T00:00:00+08:00",
        metrics=metrics,
        datasets=datasets if datasets is not None else {"toy_set": "v1"},
    )


def test_regression_detected():
    base = rec("a", {"recall": 0.80})
    head = rec("b", {"recall": 0.70})
    result = compare(base, head, SPEC)
    assert [d.verdict for d in result.deltas] == ["regressed"]
    assert result.exit_code == EXIT_REGRESSION


def test_improvement_detected():
    result = compare(rec("a", {"recall": 0.7}), rec("b", {"recall": 0.9}), SPEC)
    assert result.deltas[0].verdict == "improved"
    assert result.exit_code == EXIT_OK


def test_lower_is_better_metric():
    """不可判定占比从 5% 涨到 20% 是退化——方向声明生效，没有默认成「越大越好」。"""
    result = compare(
        rec("a", {"undeterminable_rate": 0.05}),
        rec("b", {"undeterminable_rate": 0.20}),
        SPEC,
    )
    assert result.deltas[0].verdict == "regressed"


def test_lower_is_better_metric_improves():
    result = compare(
        rec("a", {"p95": 120.0}),
        rec("b", {"p95": 80.0}),
        SPEC,
    )
    assert result.deltas[0].verdict == "improved"


def test_tolerance_suppresses_noise():
    """0.0005 的抖动不该被报成退化，否则工具会被人在噪声里关掉。"""
    result = compare(rec("a", {"recall": 0.8000}), rec("b", {"recall": 0.7995}), SPEC, tolerance=0.001)
    assert result.deltas[0].verdict == "unchanged"
    assert result.exit_code == EXIT_OK


def test_tolerance_can_be_zero():
    result = compare(rec("a", {"recall": 0.8000}), rec("b", {"recall": 0.7995}), SPEC, tolerance=0.0)
    assert result.deltas[0].verdict == "regressed"


def test_new_and_dropped_metrics():
    result = compare(rec("a", {"recall": 0.8}), rec("b", {"recall": 0.8, "p95": 90.0}), SPEC)
    verdicts = {d.key: d.verdict for d in result.deltas}
    assert verdicts["p95"] == "new"

    result = compare(rec("a", {"recall": 0.8, "p95": 90.0}), rec("b", {"recall": 0.8}), SPEC)
    verdicts = {d.key: d.verdict for d in result.deltas}
    assert verdicts["p95"] == "dropped"


def test_undeclared_metric_is_unknown_not_guessed():
    """未声明方向的指标不许猜。猜错一次就会把退化报成改进。"""
    result = compare(rec("a", {"mystery": 1.0}), rec("b", {"mystery": 2.0}), SPEC)
    assert result.deltas[0].verdict == "unknown"
    assert any("未声明方向" in w for w in result.warnings)


def test_different_task_is_incomparable():
    result = compare(rec("a", {"recall": 0.8}), rec("b", {"recall": 0.9}, task="other"), SPEC)
    assert result.comparable is False
    assert result.exit_code == EXIT_INCOMPARABLE


def test_different_dataset_is_incomparable():
    """数据换了还照样比，是这类工具最危险的失败方式——必须挡住。"""
    base = rec("a", {"recall": 0.8}, datasets={"toy_set": "n=100"})
    head = rec("b", {"recall": 0.9}, datasets={"toy_set": "n=300"})
    result = compare(base, head, SPEC)
    assert result.comparable is False
    assert result.exit_code == EXIT_INCOMPARABLE
    assert any("数据集" in w for w in result.warnings)


def test_formatting():
    assert format_value(0.696, "ratio") == "69.6%"
    assert format_value(90.0, "ms") == "90ms"
    assert format_value(None, "ratio") == "—"
    assert format_diff(0.02, "ratio") == "+2.0pp"
    assert format_diff(-0.355, "ratio") == "-35.5pp"


def test_render_smoke():
    """render 必须单独测。

    真实教训：compare 的逻辑全测过了，但 render 里把 `format_diff` 写成了
    `format_delta`——CLI 一跑就 NameError。**逻辑测了、表现层没测，bug 就长在这里。**
    """
    result = compare(
        rec("a", {"recall": 0.80, "undeterminable_rate": 0.05}),
        rec("b", {"recall": 0.70, "undeterminable_rate": 0.20}),
        SPEC,
    )
    text = render(result)
    assert "退化" in text
    assert "-10.0pp" in text
    assert "退出码 1" in text
    assert result.base.run_id in text and result.head.run_id in text


def test_render_incomparable_notice():
    base = rec("a", {"recall": 0.8}, datasets={"toy_set": "n=100"})
    head = rec("b", {"recall": 0.9}, datasets={"toy_set": "n=300"})
    text = render(compare(base, head, SPEC))
    assert "不可比对" in text
    assert "退出码 2" in text
