"""测试中台入口。

    python scripts/testbench.py run labels --mode fields
    python scripts/testbench.py run labels --mode text
    python scripts/testbench.py list
    python scripts/testbench.py compare <base_run_id> <head_run_id>

`compare` 退出码：0 无退化 / 1 有退化 / 2 不可比（任务或数据指纹不同）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in sorted((ROOT / "packages").glob("*/src")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from mc_testbench.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
