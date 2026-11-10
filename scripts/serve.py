"""本地起服务，省得记 PYTHONPATH。

    python scripts/serve.py medireg          # P1，:8011
    python scripts/serve.py labelguard       # P2，:8012
    python scripts/serve.py dossier          # P3，:8013
    python scripts/serve.py medireg --reload

为什么需要这个脚本：monorepo 里 `mc_medireg` 依赖兄弟目录的 `mc_domain`，
uvicorn 的 `--app-dir` 只能加一个路径，直接跑必然 ImportError。
正式开发用 `uv sync` 装好即可；不装依赖时用这个脚本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for src in sorted((ROOT / "packages").glob("*/src")):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from mc_domain.env import load_dotenv  # noqa: E402

# 显式指定仓库根的 .env：不依赖"从哪个目录启动"
loaded = load_dotenv(ROOT / ".env")

SERVICES: dict[str, tuple[str, int, str]] = {
    "medireg": ("mc_medireg.main:app", 8011, "P1 MediReg 械规通"),
    "labelguard": ("mc_labelguard.main:app", 8012, "P2 LabelGuard 标签哨兵"),
    "dossier": ("mc_dossier.main:app", 8013, "P3 DossierReview 卷宗审查"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="起一个 MedCompliance 服务")
    parser.add_argument("service", choices=sorted(SERVICES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="默认按服务分配")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    target, default_port, label = SERVICES[args.service]

    # 延后 import：uvicorn 只有在路径就绪后才可加载依赖里含 mc_domain 的模块
    import uvicorn

    port = args.port or default_port
    print(f"[serve] {label} → http://{args.host}:{port}  (docs: /docs)")
    if loaded:
        print(f"[serve] 已加载环境文件：{loaded}")
    else:
        print("[serve] 未找到 .env，使用默认配置（heuristic 抽取 + 合成目录）")
    uvicorn.run(target, host=args.host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
