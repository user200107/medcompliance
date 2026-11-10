"""让 `pytest` 在仓库根目录直接可跑，不需要先 uv sync / pip install -e。

monorepo 的 src-layout 如果不做这一步，每个包都要装一遍才能互相 import；
业余项目里这层摩擦会直接导致"懒得跑测试"。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for src in sorted((ROOT / "packages").glob("*/src")):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
