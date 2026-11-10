"""HTTP 接口。

三条路由对应三种使用方式：
- `POST /api/v1/classify`          只给一段描述，服务内部做槽位抽取
- `POST /api/v1/classify/profile`  前端答完追问后，带着完整画像回来
- `GET  /api/v1/catalog/{code}`    查某个分类编码的定义、排除说明、品名举例
"""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Request, status
from mc_domain.decision import available_presets, get_preference
from mc_domain.schemas import ClassificationCodeRead, ClassificationResult, ProductProfile
from pydantic import BaseModel, Field

from mc_medireg.normalizer import NormalizationFailed

logger = logging.getLogger(__name__)
router = APIRouter()


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=1, max_length=5000, description="产品自由文本描述")
    as_of: date | None = Field(default=None, description="按哪一天的法律状态判定，默认今天")
    risk: str | None = Field(
        default=None,
        description="风险偏好预设：strict / balanced / risk_averse / max_recall_iii，默认 balanced",
    )


class ClassifyProfileRequest(BaseModel):
    """前端答完追问后的回填。允许槽位仍为空——那会继续追问，而不是报错。"""

    profile: ProductProfile
    as_of: date | None = None
    risk: str | None = None


@router.get("/healthz", summary="存活与目录状态")
async def healthz(request: Request) -> dict[str, object]:
    state = request.app.state
    return {
        "status": "ok",
        "service": state.settings.service_name,
        "catalog_version": state.catalog.version,
        "catalog_nodes": state.catalog.node_count,
        "classifiable_nodes": state.catalog.classifiable_count,
        "normalizer": state.settings.normalizer,
    }


@router.get("/api/v1/risk-presets", summary="列出可选的风险偏好")
async def list_risk_presets() -> dict[str, object]:
    """把取舍策略暴露给前端——选 Ⅱ 类还是 Ⅲ 类，本该由用户按场景决定。"""
    return {
        "default": "balanced",
        "presets": available_presets(),
        "note": "预设名可直接传给 /api/v1/classify 的 risk 字段。",
    }


def _resolve_risk(risk: str | None) -> str | None:
    if risk is None:
        return None
    try:
        get_preference(risk)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_risk_preference", "message": str(exc)},
        ) from exc
    return risk


@router.post(
    "/api/v1/classify",
    response_model=ClassificationResult,
    summary="按描述分类（内部做槽位抽取）",
)
async def classify(request: Request, body: ClassifyRequest) -> ClassificationResult:
    pipeline = request.app.state.pipeline
    risk = _resolve_risk(body.risk)
    try:
        return await pipeline.classify(body.description, as_of=body.as_of, risk=risk)
    except NormalizationFailed as exc:
        # 抽取失败必须是 5xx，绝不能返回一个「看起来成功但空」的结果
        logger.warning("槽位抽取失败：%s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "normalization_failed", "message": str(exc)},
        ) from exc


@router.post(
    "/api/v1/classify/profile",
    response_model=ClassificationResult,
    summary="按已结构化画像分类（追问回填）",
)
async def classify_with_profile(
    request: Request, body: ClassifyProfileRequest
) -> ClassificationResult:
    pipeline = request.app.state.pipeline
    risk = _resolve_risk(body.risk)
    profile = body.profile
    if not profile.raw_text:
        profile = profile.model_copy(update={"raw_text": _render_raw_text(profile)})
    return await pipeline.classify(
        profile.raw_text, as_of=body.as_of, profile=profile, risk=risk
    )


@router.get(
    "/api/v1/catalog/{code}",
    response_model=ClassificationCodeRead,
    summary="查看分类目录节点",
)
async def get_catalog_node(request: Request, code: str) -> ClassificationCodeRead:
    tree = request.app.state.catalog.tree
    node = tree.maybe_get(code)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "catalog_node_not_found", "message": f"目录中没有编码 {code}"},
        )
    return ClassificationCodeRead(
        id=0,
        catalog_version=node.catalog_version,
        code=node.code,
        level=node.level,
        name=node.name,
        name_path=node.name_path,
        management_class=node.management_class,
        exclusion_note=node.exclusion_note,
        examples=node.examples,
    )


def _render_raw_text(profile: ProductProfile) -> str:
    """回填的画像没有原文时，拼一段可读文本给检索层用。"""
    parts = [
        profile.intended_use or "",
        profile.name or "",
        profile.principle or "",
        profile.structure_composition or "",
        " ".join(profile.materials),
    ]
    return " ".join(p for p in parts if p).strip() or "（未提供描述）"
