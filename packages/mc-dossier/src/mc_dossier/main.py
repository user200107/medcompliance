"""应用工厂。"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mc_domain.env import cors_origins

from mc_dossier.api import router
from mc_dossier.version import __version__


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    app = FastAPI(
        title="DossierReview 卷宗审查",
        version=__version__,
        description="医疗器械申报资料审查：章节完整性 + 跨文档属性一致性（含归一化）",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
