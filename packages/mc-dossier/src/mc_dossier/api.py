"""P3 的 HTTP 接口。

**刻意做成无状态**：卷宗审查产出的是"问题清单"，供审核员逐条处理，
不需要像 P2 那样把结论冻结成对外报告。所以这里没有草稿/确认门禁——
这不是偷懒，是两个应用的产品定位不同：

- P2 标签审核：结论会被拿去整改甚至作为合规证据 → 必须人署名确认、不可变快照
- P3 卷宗审查：产出是待办清单 → 每次请求按当时文档重新算即可

把这条差异讲清楚，比给两个应用套同一套门禁更有说服力。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from mc_dossier.consistency import (
    REQUIRED_DOCS,
    extract_fields_from_text,
    review_dossier,
)
from mc_dossier.schemas import DOC_LABELS, Attribute, DocKind, Document, DossierReviewResult

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentInput(BaseModel):
    doc_id: str
    kind: DocKind
    filename: str = ""
    fields: dict[str, str] = Field(
        default_factory=dict, description="已抽取的属性；为空时尝试从 text 正则抽取"
    )
    text: str = ""


class ReviewRequest(BaseModel):
    docs: list[DocumentInput] = Field(min_length=1)
    dossier_id: str = "dossier"
    auto_extract: bool = Field(
        default=True, description="fields 为空时是否用正则从 text 抽取"
    )


@router.get("/healthz", summary="存活与规则状态")
async def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "mc-dossier",
        "implemented": True,
        "required_docs": [k.value for k in REQUIRED_DOCS],
        "cross_checked_attributes": [a.value for a in Attribute],
        "stateless": True,
    }


@router.get("/api/v1/dossiers/requirements", summary="列出应有章节与核对项")
async def list_requirements() -> dict[str, object]:
    return {
        "required_docs": [{"kind": k.value, "label": DOC_LABELS[k]} for k in REQUIRED_DOCS],
        "attributes": [
            {"key": a.value, "label": __import__("mc_dossier.schemas", fromlist=["ATTRIBUTE_LABELS"]).ATTRIBUTE_LABELS[a]}
            for a in Attribute
        ],
        "note": "比对范围仅限有确定值的结构化属性；预期用途等自由文本不纳入，避免制造噪音。",
    }


@router.post(
    "/api/v1/dossiers/review",
    response_model=DossierReviewResult,
    summary="审查一套申报资料（缺项 + 跨文档冲突）",
)
async def review(request: ReviewRequest) -> DossierReviewResult:
    docs: list[Document] = []
    for item in request.docs:
        fields = dict(item.fields)
        if not fields and request.auto_extract and item.text:
            fields = extract_fields_from_text(item.kind, item.text)
        docs.append(
            Document(
                doc_id=item.doc_id,
                kind=item.kind,
                filename=item.filename or f"{item.kind.value}.txt",
                fields=fields,
                text=item.text,
            )
        )

    try:
        result = review_dossier(docs, dossier_id=request.dossier_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "empty_dossier", "message": str(exc)},
        ) from exc

    logger.info(
        "卷宗审查 dossier=%s 文档=%d 冲突=%d 缺项=%d",
        result.dossier_id,
        len(docs),
        result.conflicts,
        len(result.missing_docs),
    )
    return result
