"""审核草稿的存储与发布门禁。

## 为什么 AI 结论不能是最终产物

标签审核的结论会被拿去整改、甚至作为合规证据。所以它必须满足两条：
1. **AI 只出草稿**，必须经审核员确认才成为正式报告；
2. **确认后的报告不可变**，任何人再改都要产生新版本（revision 递增）。

这两条把"AI 不能负最终责任"从一句免责声明变成了一段代码。

## 状态机

    pending_review ──confirm──> confirmed（不可变快照）
          │
          └── 可反复复核（revision CAS：拿旧 revision 提交会 409）

## 存储

内存字典。**这是刻意的**：P2 目前是单进程演示，没有多租户与持久化需求，
为了一个作品集去接 MySQL + 迁移脚本是过度设计。
`DraftStore` 的接口按"可替换成数据库实现"设计（confirm 用 revision 做 CAS，
天然对应数据库的行锁 + 版本列），换实现时调用方不用改。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from mc_labelguard.schemas import LabelReviewResult


class DraftStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"


class DraftNotFound(KeyError):
    pass


class RevisionConflict(RuntimeError):
    """提交的 revision 不是最新版——说明别人已经复核过了。"""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"报告版本已变化：你基于 r{expected}，当前是 r{actual}，请刷新后重试")
        self.expected = expected
        self.actual = actual


@dataclass
class LabelDraft:
    draft_id: str
    review: LabelReviewResult
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    status: DraftStatus = DraftStatus.PENDING_REVIEW
    revision: int = 1
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    note: str | None = None
    snapshot: dict[str, Any] | None = None
    """确认时冻结的完整快照。之后任何改动都只能新建版本，不能改它。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "status": self.status.value,
            "revision": self.revision,
            "source": self.source,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at.isoformat(timespec="seconds") if self.reviewed_at else None,
            "note": self.note,
            "review": self.review.model_dump(mode="json"),
            "snapshot": self.snapshot,
        }

    def to_ai_draft(self) -> dict[str, Any]:
        """给前端的"待复核"视图：不含确认信息，且明确标注这是草稿。"""
        return {
            "draft_id": self.draft_id,
            "status": self.status.value,
            "revision": self.revision,
            "is_draft": True,
            "review": self.review.model_dump(mode="json"),
        }


class DraftStore:
    def __init__(self) -> None:
        self._drafts: dict[str, LabelDraft] = {}

    def create(self, review: LabelReviewResult, *, source: str) -> LabelDraft:
        draft = LabelDraft(draft_id=uuid.uuid4().hex[:12], review=review, source=source)
        self._drafts[draft.draft_id] = draft
        return draft

    def get(self, draft_id: str) -> LabelDraft:
        draft = self._drafts.get(draft_id)
        if draft is None:
            raise DraftNotFound(draft_id)
        return draft

    def confirm(
        self,
        draft_id: str,
        *,
        reviewer: str,
        expected_revision: int,
        note: str | None = None,
    ) -> LabelDraft:
        if not reviewer or not reviewer.strip():
            # 没有署名就不能发布——否则等于"AI 直接出报告"
            raise ValueError("确认发布必须提供审核员姓名")

        draft = self.get(draft_id)
        if draft.status is DraftStatus.CONFIRMED:
            raise RevisionConflict(expected_revision, draft.revision)
        if draft.revision != expected_revision:
            raise RevisionConflict(expected_revision, draft.revision)

        draft.status = DraftStatus.CONFIRMED
        draft.reviewer = reviewer.strip()
        draft.reviewed_at = datetime.now()
        draft.note = note
        draft.revision += 1
        draft.snapshot = {
            "confirmed_at": draft.reviewed_at.isoformat(timespec="seconds"),
            "reviewer": draft.reviewer,
            "note": note,
            "review": draft.review.model_dump(mode="json"),
            "immutable": True,
        }
        return draft

    def clear(self) -> None:
        """测试用。"""
        self._drafts.clear()

    def __len__(self) -> int:
        return len(self._drafts)


__all__ = [
    "DraftNotFound",
    "DraftStatus",
    "DraftStore",
    "LabelDraft",
    "RevisionConflict",
]
