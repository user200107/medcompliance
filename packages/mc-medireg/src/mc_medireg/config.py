"""配置：只用环境变量，不引入 pydantic-settings。

理由：一个人维护的项目，少一个依赖就少一处升级折腾；
配置项一共就几个，dataclass + os.environ 足够，而且启动时会打印生效值（密钥除外）。

## 分类目录的选择顺序（这一条是硬需求）

1. `MC_CATALOG_CSV` 显式指定 → 用它；
2. 否则**向上搜索 `data/raw/catalog_real.csv`**，找到就用；
3. 都没有 → 退回内置的合成目录（8 个节点）。

为什么必须自动找：真实目录在 `data/raw/`（已 gitignore），clone 下来的人拿到的是合成目录。
这本身没问题，但**演示时用 8 个节点的假目录很难看**——
`/healthz` 会明说 `catalog_nodes: 8`，而 README 写着 1,917，一眼就不一致。
靠"记得设环境变量"是不可靠的，所以默认就该优先真实目录。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from mc_domain.env import load_dotenv

logger = logging.getLogger(__name__)

#: 真实目录的约定位置（相对仓库根）。抓取脚本默认就写到这里。
REAL_CATALOG_RELATIVE = Path("data") / "raw" / "catalog_real.csv"


def _search_upwards(start: Path) -> Path | None:
    for candidate_dir in (start, *start.parents):
        candidate = candidate_dir / REAL_CATALOG_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def _default_catalog_csv() -> Path:
    found = _search_upwards(Path(__file__).resolve().parent) or _search_upwards(Path.cwd())
    if found is not None:
        return found
    return Path(str(resources.files("mc_domain").joinpath("data/catalog_sample.csv")))


def _default_catalog_version(catalog_csv: Path) -> str:
    """版本标签跟着目录来源走，避免"真实目录配了个 sample 版本号"这种自相矛盾。"""
    explicit = os.getenv("MC_CATALOG_VERSION", "").strip()
    if explicit:
        return explicit
    return "real-catalog" if catalog_csv.name == "catalog_real.csv" else "sample-synthetic"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    service_name: str = "mc-medireg"

    catalog_csv: Path = field(
        default_factory=lambda: Path(os.getenv("MC_CATALOG_CSV") or _default_catalog_csv())
    )
    catalog_version: str = field(default="")

    #: "heuristic"（离线降级）或 "llm"
    normalizer: str = field(default_factory=lambda: os.getenv("MC_NORMALIZER", "heuristic").strip().lower())

    llm_base_url: str = field(default_factory=lambda: os.getenv("MC_LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("MC_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("MC_LLM_MODEL", "qwen-plus"))
    llm_timeout_s: int = field(default_factory=lambda: _int("MC_LLM_TIMEOUT_S", 30))
    llm_max_retries: int = field(default_factory=lambda: _int("MC_LLM_MAX_RETRIES", 2))

    debug: bool = field(default_factory=lambda: _bool("MC_DEBUG", False))

    def __post_init__(self) -> None:
        # catalog_version 依赖 catalog_csv，只能在这里补
        if not self.catalog_version:
            object.__setattr__(self, "catalog_version", _default_catalog_version(self.catalog_csv))

    def describe(self) -> dict[str, object]:
        """启动日志用。密钥只报是否存在，绝不输出内容。"""
        return {
            "service": self.service_name,
            "catalog_csv": str(self.catalog_csv),
            "catalog_version": self.catalog_version,
            "normalizer": self.normalizer,
            "llm_model": self.llm_model if self.normalizer == "llm" else "-",
            "llm_api_key_set": bool(self.llm_api_key),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.catalog_csv.exists():
            problems.append(f"分类目录文件不存在：{self.catalog_csv}")
        if self.normalizer not in {"heuristic", "llm"}:
            problems.append(f"MC_NORMALIZER 只能是 heuristic 或 llm，实际：{self.normalizer}")
        if self.normalizer == "llm":
            if not self.llm_base_url:
                problems.append("normalizer=llm 时必须设置 MC_LLM_BASE_URL")
            if not self.llm_api_key:
                problems.append("normalizer=llm 时必须设置 MC_LLM_API_KEY")
        return problems


def get_settings() -> Settings:
    # 先加载仓库根的 .env；已存在的环境变量优先于文件（便于临时覆盖）
    loaded = load_dotenv()
    settings = Settings()
    if loaded:
        logger.debug("已加载环境文件：%s", loaded)
    return settings
