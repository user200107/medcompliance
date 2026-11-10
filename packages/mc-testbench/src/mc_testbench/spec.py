"""任务与指标的元数据。

## 为什么每个指标必须声明方向

`undeterminable_rate` 越低越好，`recall` 越高越好。
如果工具默认"数值变大就是变好"，它会把这个最常见的看板指标报成**改进**——
这是这类工具最容易犯、也最快失去信任的错误。

所以方向不是猜的，是**声明**的；未声明的指标一律判 `unknown`，不猜。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: ratio 按百分比显示（0.696 → 69.6%），count 显示整数，ms 显示毫秒
Unit = Literal["ratio", "count", "ms"]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: Unit = "ratio"
    higher_is_better: bool = True

    def is_better(self, before: float, after: float) -> bool:
        return after > before if self.higher_is_better else after < before


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    metrics: tuple[MetricSpec, ...]
    datasets: tuple[str, ...] = ()

    def metric(self, key: str) -> MetricSpec | None:
        for spec in self.metrics:
            if spec.key == key:
                return spec
        return None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return tuple(m.key for m in self.metrics)
