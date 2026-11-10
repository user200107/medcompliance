"""P2 的 HTTP 接口。

工作流与真实场景对齐：

    POST /api/v1/labels/review          送审 → 返回**草稿**（明确 is_draft: true）
    GET  /api/v1/labels/drafts/{id}     复核人查看草稿
    POST /api/v1/labels/drafts/{id}/confirm   审核员署名确认 → 生成不可变快照

`review` 支持两种输入：`text`（文本标签，走正则抽取）或 `fields`
（已结构化的抽取结果，供未来接视觉模型时直接用）。**视觉还没接**，
接口层先留好位置，避免以后改客户端。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from mc_domain.enums import RequirementKey
from pydantic import BaseModel, Field

from mc_labelguard.checklist import DEFAULT_MIN_CONFIDENCE, RULES, check_label
from mc_labelguard.extract import extract_from_text
from mc_labelguard.schemas import LabelExtraction, LabelReviewResult
from mc_labelguard.store import DraftNotFound, DraftStore, RevisionConflict

logger = logging.getLogger(__name__)
router = APIRouter()


class ReviewRequest(BaseModel):
    text: str | None = Field(default=None, description="标签文本（走正则抽取）")
    fields: LabelExtraction | None = Field(default=None, description="已结构化的抽取结果")
    min_confidence: float = Field(default=DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)


class ConfirmRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64, description="审核员姓名，必须署名")
    revision: int = Field(ge=1, description="你复核时看到的版本号")
    note: str | None = Field(default=None, max_length=1000)


class ReviewResponse(BaseModel):
    draft_id: str
    status: str
    revision: int
    is_draft: bool
    review: LabelReviewResult


@router.get("/healthz", summary="存活与规则表状态")
async def healthz(request: Request) -> dict[str, object]:
    store: DraftStore = request.app.state.store
    return {
        "status": "ok",
        "service": "mc-labelguard",
        "implemented": True,
        "checklist_rules": len(RULES),
        "requirement_keys": len(RequirementKey),
        "drafts_in_memory": len(store),
        "vision": "not_implemented（当前只支持 text / fields 输入）",
    }


@router.get("/api/v1/labels/checklist", summary="列出必标项规则")
async def list_checklist() -> dict[str, object]:
    """把规则表直接暴露出去——它是这个产品的核心资产，不该藏在代码里。"""
    return {
        "count": len(RULES),
        "items": [
            {
                "key": r.key.value,
                "label": r.label,
                "clause": r.clause,
                "kind": r.kind,
                "required": r.required,
                "conditional": r.applies_when is not None,
                "condition": r.condition_hint or None,
                "pattern_hint": r.pattern_hint or None,
            }
            for r in RULES
        ],
    }


@router.post(
    "/api/v1/labels/review",
    response_model=ReviewResponse,
    summary="送审一张标签（返回草稿，须审核员确认）",
)
async def review(request: Request, body: ReviewRequest) -> ReviewResponse:
    if body.fields is None and not (body.text or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "empty_input", "message": "必须提供 text 或 fields"},
        )

    extraction = body.fields or extract_from_text(body.text or "")
    result = check_label(extraction, min_confidence=body.min_confidence)

    store: DraftStore = request.app.state.store
    draft = store.create(result, source="text" if body.fields is None else "fields")
    logger.info(
        "标签送审 draft=%s 风险=%s 违规=%d", draft.draft_id, result.risk_level.value, result.violations
    )
    return ReviewResponse(**draft.to_ai_draft())


@router.get("/api/v1/labels/drafts/{draft_id}", summary="查看草稿或已确认报告")
async def get_draft(request: Request, draft_id: str) -> dict[str, object]:
    store: DraftStore = request.app.state.store
    try:
        return store.get(draft_id).to_dict()
    except DraftNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "draft_not_found", "message": f"草稿 {draft_id} 不存在"},
        ) from None


@router.post(
    "/api/v1/labels/drafts/{draft_id}/confirm",
    summary="审核员署名确认，生成不可变报告",
)
async def confirm(request: Request, draft_id: str, body: ConfirmRequest) -> dict[str, object]:
    store: DraftStore = request.app.state.store
    try:
        draft = store.confirm(
            draft_id,
            reviewer=body.reviewer,
            expected_revision=body.revision,
            note=body.note,
        )
    except DraftNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "draft_not_found", "message": f"草稿 {draft_id} 不存在"},
        ) from None
    except RevisionConflict as exc:
        # 409：别人已经复核过，你基于旧版本的操作不能覆盖
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "revision_conflict",
                "message": str(exc),
                "current_revision": exc.actual,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_reviewer", "message": str(exc)},
        ) from exc

    return draft.to_dict()
