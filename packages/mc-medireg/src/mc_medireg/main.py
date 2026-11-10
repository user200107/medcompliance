"""应用工厂。

CORS 默认只放行本地演示页（`apps/web`，:5180），不放开 `*`——
三个服务都带业务数据，通配来源等于任何网页都能从浏览器里调它们。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mc_domain import ClassificationPipeline
from mc_domain.env import cors_origins

from mc_medireg.api import router
from mc_medireg.catalog import load_catalog
from mc_medireg.config import Settings, get_settings
from mc_medireg.normalizer import build_normalizer
from mc_medireg.version import __version__

logger = logging.getLogger(__name__)


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    problems = settings.validate()
    if problems:
        raise RuntimeError("配置校验未通过：" + "；".join(problems))

    app.state.catalog = load_catalog(settings.catalog_csv, settings.catalog_version)
    normalizer = build_normalizer(
        settings.normalizer,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    app.state.pipeline = ClassificationPipeline(
        tree=app.state.catalog.tree,
        retriever=app.state.catalog.retriever,
        normalizer=normalizer,
    )
    logger.info("启动完成：%s", settings.describe())
    yield
    logger.info("服务关闭")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.debug)

    app = FastAPI(
        title="MediReg 械规通",
        version=__version__,
        description="医疗器械注册路径知识引擎：分类判定 + 法规时效 + 临床评价路径",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
