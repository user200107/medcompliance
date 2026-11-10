"""极简 .env 加载器。

不引 python-dotenv：这个项目只需要"读 KEY=VALUE"，20 行够了。
一个人维护的项目，少一个依赖就少一处升级折腾。

规则（与常见 dotenv 一致）：
- 不存在 `.env` 就静默跳过，不报错；
- **已存在的环境变量优先**，`.env` 不覆盖它们——这样 `MC_DEBUG=true python ...` 能临时压过文件；
- 支持 `#` 注释、空行、`KEY=VALUE`，值两侧的引号会被去掉；
- 不支持的语法（如 `export`）按普通键处理，不抛异常。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"


def find_dotenv(start: Path | None = None, *, max_levels: int = 6) -> Path | None:
    """从 start 向上找 .env。默认从当前工作目录开始。"""
    cursor = (start or Path.cwd()).resolve()
    for _ in range(max_levels):
        candidate = cursor / ENV_FILENAME
        if candidate.is_file():
            return candidate
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """把 .env 内容解析成字典。纯函数，便于单测。"""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Path | None = None, *, override: bool = False) -> Path | None:
    """加载 .env 到 os.environ。返回实际加载的文件路径（没找到则 None）。"""
    env_path = path or find_dotenv()
    if env_path is None or not env_path.is_file():
        return None

    for key, value in parse_dotenv(env_path.read_text(encoding="utf-8")).items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def http_trust_env() -> bool:
    """httpx 是否读取 HTTP_PROXY / NO_PROXY 等环境变量。

    默认 **False**，原因是一个真实踩到的坑：某些 Windows 环境的 `NO_PROXY`
    含有 httpx 的 URL 解析器无法处理的条目（如 `[::1]`），
    会导致 `httpx.Client()` **在构造阶段**就抛 InvalidURL —— 还没发出任何请求。

    关掉 trust_env 后不再让 httpx 自己解析这些变量，改由 `http_proxy_url()`
    显式取代理地址（忽略 NO_PROXY）。确实需要整套 env 行为时设 `MC_HTTP_TRUST_ENV=true`。
    """
    return os.getenv("MC_HTTP_TRUST_ENV", "").strip().lower() in {"1", "true", "yes", "on"}


def http_proxy_url() -> str | None:
    """取一个可用的代理地址。

    **刻意不读 NO_PROXY**：它只影响"哪些目标不走代理"，而我们无法安全地把它交给
    httpx 解析（见 `http_trust_env` 的说明）。对本项目的用途（访问百炼 API）
    直接走代理是正确且可预期的。
    """
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def http_client_kwargs(timeout: float) -> dict[str, object]:
    """给 httpx.Client / httpx.AsyncClient 的统一参数。

    同步与异步客户端共用同一套逻辑，避免两处各写一遍、日后走偏。
    """
    kwargs: dict[str, object] = {"timeout": timeout, "trust_env": http_trust_env()}
    if not kwargs["trust_env"]:
        proxy = http_proxy_url()
        if proxy:
            kwargs["proxy"] = proxy
    return kwargs


def cors_origins() -> list[str]:
    """允许的前端来源。

    **默认只放行本地演示页**，而不是 `*`：
    三个服务都带业务数据，通配来源等于谁都能从浏览器里调。
    需要别的来源时设 `MC_CORS_ORIGINS=https://a.example,https://b.example`。

    另外不开 `allow_credentials`：本套件用 Bearer/无鉴权，不用 Cookie；
    而 `allow_origins=["*"] + allow_credentials=True` 是浏览器会直接拒绝的组合，
    很多项目踩过这个坑。
    """
    raw = os.getenv("MC_CORS_ORIGINS", "").strip()
    if not raw:
        return ["http://127.0.0.1:5180", "http://localhost:5180"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


__all__ = [
    "ENV_FILENAME",
    "cors_origins",
    "find_dotenv",
    "http_client_kwargs",
    "http_proxy_url",
    "http_trust_env",
    "load_dotenv",
    "parse_dotenv",
]
