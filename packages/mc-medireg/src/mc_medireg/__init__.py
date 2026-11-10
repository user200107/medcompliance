"""P1 MediReg 械规通 —— 医疗器械注册路径知识引擎。

对外只暴露 `create_app()`；uvicorn 指向 `mc_medireg.main:app`。
"""

from __future__ import annotations

from mc_medireg.main import create_app
from mc_medireg.version import __version__

__all__ = ["create_app", "__version__"]
