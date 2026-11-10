"""静态托管 `apps/web`（演示用）。

    python scripts/serve_web.py            # http://127.0.0.1:5180
    python scripts/serve_web.py --port 8080

刻意用标准库：前端是零构建的单文件，为它引一个静态服务器框架没有意义。
生产环境这份静态资源应由 nginx 托管（见 apps/web/README.md）。
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "apps" / "web"


class _Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # 演示页会频繁改动，禁掉缓存省得每次手动强刷
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        print(f"[web] {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="托管演示前端")
    parser.add_argument("--port", type=int, default=5180)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not (WEB_DIR / "index.html").is_file():
        raise SystemExit(f"找不到 {WEB_DIR / 'index.html'}")

    handler = functools.partial(_Handler, directory=str(WEB_DIR))
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"[web] http://{args.host}:{args.port}  → {WEB_DIR}")
        print("[web] 需要同时启动三个服务：")
        print("      python scripts/serve.py medireg")
        print("      python scripts/serve.py labelguard")
        print("      python scripts/serve.py dossier")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[web] 已停止")


if __name__ == "__main__":
    main()
