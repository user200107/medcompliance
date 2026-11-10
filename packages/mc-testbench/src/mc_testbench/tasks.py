"""任务适配器：把已有的评测脚本接成可记录的 run。

**刻意不重写评测逻辑。** 评测脚本是资产，中台只负责"记录与对比"——
它通过 importlib 加载 `scripts/` 下的脚本并调用其 `evaluate()`，
这样脚本继续能单独跑，中台也拿到了结构化指标。

同理，数据指纹对合成集不需要哈希：`synthetic:labels:n=300,seed=20261110`
本身就是确定的，换 n 或 seed 会自然变化——这足以发现"两次评测不是在同一个集上跑"。
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from mc_testbench.spec import MetricSpec, TaskSpec


@dataclass
class RunOutcome:
    metrics: dict[str, float]
    config: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, str] = field(default_factory=dict)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / "eval_labels.py").is_file():
            return parent
    raise RuntimeError("找不到仓库根：未发现 scripts/eval_labels.py")


def _load_script(stem: str) -> ModuleType:
    root = _repo_root()
    for package_src in sorted((root / "packages").glob("*/src")):
        if str(package_src) not in sys.path:
            sys.path.insert(0, str(package_src))
    path = root / "scripts" / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"找不到脚本：{path}")
    spec = importlib.util.spec_from_file_location(f"_tb_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flatten(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {k: float(metrics[k]) for k in keys if isinstance(metrics.get(k), (int, float))}


# ---------------------------------------------------------------------------
# 标签合规审核
# ---------------------------------------------------------------------------


def run_labels(mode: str = "fields", n: int = 300, seed: int = 20261110) -> RunOutcome:
    module = _load_script("eval_labels")
    samples = module.generate(n, seed=seed)
    raw = module.evaluate(samples, mode)
    return RunOutcome(
        metrics=_flatten(raw, LABELS.metric_keys),
        config={"mode": mode, "n": n, "seed": seed},
        datasets={"synthetic_labels": f"n={n};seed={seed}"},
    )


# ---------------------------------------------------------------------------
# 卷宗一致性
# ---------------------------------------------------------------------------


def run_dossier(n: int = 180, seed: int = 20261110) -> RunOutcome:
    module = _load_script("eval_dossier")
    samples = module.generate(n, seed=seed)
    raw = module.evaluate(samples)
    return RunOutcome(
        metrics=_flatten(raw, DOSSIER.metric_keys),
        config={"n": n, "seed": seed},
        datasets={"synthetic_dossier": f"n={n};seed={seed}"},
    )


LABELS = TaskSpec(
    name="labels",
    title="标签合规审核（合成集）",
    metrics=(
        MetricSpec("precision", "精确率"),
        MetricSpec("recall", "召回率"),
        MetricSpec("f1", "F1"),
        MetricSpec("clean_fp_rate", "负控假阳性", higher_is_better=False),
        MetricSpec("undeterminable_rate", "不可判定占比", higher_is_better=False),
    ),
    datasets=("synthetic_labels",),
)

DOSSIER = TaskSpec(
    name="dossier",
    title="卷宗一致性（合成集）",
    metrics=(
        MetricSpec("precision", "精确率"),
        MetricSpec("recall", "召回率"),
        MetricSpec("f1", "F1"),
        MetricSpec("clean_fp_rate", "负控假阳性", higher_is_better=False),
        MetricSpec("missing_recall", "缺项检出率"),
    ),
    datasets=("synthetic_dossier",),
)

TASKS: dict[str, TaskSpec] = {spec.name: spec for spec in (LABELS, DOSSIER)}

RUNNERS: dict[str, Callable[..., RunOutcome]] = {
    "labels": run_labels,
    "dossier": run_dossier,
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"未知任务 {name}，可选：{', '.join(TASKS)}")
    return TASKS[name]


def run(task: str, **config: Any) -> RunOutcome:
    if task not in RUNNERS:
        raise KeyError(f"任务 {task} 没有接入运行器")
    spec = get_task(task)
    allowed = {"labels": {"mode", "n", "seed"}, "dossier": {"n", "seed"}}[task]
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"任务 {task} 不接受参数：{', '.join(sorted(unknown))}")
    return RUNNERS[task](**config)
