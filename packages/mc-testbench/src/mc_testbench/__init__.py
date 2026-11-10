"""评测运行管理与回归检测。

**它解决的不是"跑评测"，是"知道有没有变差"。**

一个评测脚本能告诉你"现在 Top-1 是 69.6%"，
但它回答不了"比上次低了还是高了"、"这次改动值得合并吗"。
后两个问题才是评测存在的意义，也是这里唯一做的事：

1. 把每次评测落成一条 **运行记录**（指标 + 配置 + 数据指纹 + commit）；
2. 把两条记录**对比**，按指标方向判断改进/退化；
3. 退化时**返回非零退出码**，从而能挂进 CI 当门禁。

刻意不做的事：不做调度器、不做看板、不做分布式执行。
跑评测仍然是同步脚本，这里只负责记录与判定。
"""

from __future__ import annotations

from mc_testbench.compare import (
    EXIT_INCOMPARABLE,
    EXIT_OK,
    EXIT_REGRESSION,
    Comparison,
    Delta,
    compare,
)
from mc_testbench.record import RunRecord, list_runs, load_run, save_run
from mc_testbench.spec import MetricSpec, TaskSpec

__all__ = [
    "EXIT_INCOMPARABLE",
    "EXIT_OK",
    "EXIT_REGRESSION",
    "Comparison",
    "Delta",
    "MetricSpec",
    "RunRecord",
    "TaskSpec",
    "compare",
    "list_runs",
    "load_run",
    "save_run",
]
