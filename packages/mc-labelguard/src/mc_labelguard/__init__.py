"""P2 LabelGuard 标签哨兵 —— 标签/说明书合规视觉审核。

**当前实现范围（诚实边界）**
- ✅ 必标项规则引擎（15 条，依《医疗器械说明书和标签管理规定》第 6 号令 /
  YY/T 0466.1 / UDI 规则），三态判定 + 条款依据
- ✅ 文本抽取（正则）与结构化字段入参
- ✅ 审核草稿 → 审核员署名确认 → 不可变快照
- ⬜ **视觉抽取未接**（qwen-vl）。接口层已留 `fields` 入口，接上后下游不用改

`create_app()` 是唯一出口；uvicorn 指向 `mc_labelguard.main:app`。
"""

from __future__ import annotations

from mc_labelguard.main import create_app
from mc_labelguard.version import __version__

__all__ = ["create_app", "__version__"]
