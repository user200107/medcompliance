"""应用工厂。"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mc_domain.env import cors_origins

from mc_labelguard.api import router
from mc_labelguard.store import DraftStore
from mc_labelguard.version import __version__


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    app = FastAPI(
        title="LabelGuard 标签哨兵",
        version=__version__,
        description="医疗器械标签/说明书合规审核：必标项三态判定 + 审核员发布门禁",
    )
    app.state.store = DraftStore()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
