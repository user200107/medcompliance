"""P2 LabelGuard 的数据契约。

核心设计：**必标项是数据，不是代码。**
加一个审查项 = 在 `checklist.py` 的规则表里加一行 + 在
`mc_domain.enums.RequirementKey` 加一个枚举值，判定引擎不动。

三态判定：
- `compliant`     合规
- `violation`     **明确**不合规（缺项、格式错、与主数据不符）
- `undeterminable` 不可判定（图片不清、字段未提取、置信度不足）

第三态是关键：**识别失败不许变成"违规"，也不许变成"合规"。**
它必须单独存在，并在报告里被统计（`undeterminable_rate`）。
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from mc_domain.enums import RequirementKey
from pydantic import BaseModel, ConfigDict, Field


class ItemStatus(str, Enum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    UNDETERMINABLE = "undeterminable"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LabelExtraction(BaseModel):
    """从标签图/文本里抽出来的结构化字段。

    字段名与 `RequirementKey` 一一对应（去掉 `label.` 前缀），
    这样规则表可以直接按 key 取值，不需要中间映射表。

    **无菌 / 一次性使用必须拆成两个字段**，它们回答的是不同问题：
    - `is_sterile` / `is_single_use` —— **产品属性**，决定该项是否适用；
    - `sterile_mark` / `single_use_mark` —— **标签上有没有那个符号**，这是被检查的对象。

    混成一个字段会导致一个真实漏洞：无菌产品漏标符号时，因为字段有值而判"已标注"。
    """

    model_config = ConfigDict(protected_namespaces=())

    product_name: str | None = None
    model_spec: str | None = None
    registrant_name: str | None = None
    registrant_address: str | None = None
    registrant_contact: str | None = None
    cert_no: str | None = None
    production_license: str | None = None
    manufacture_date: date | None = None
    expiry_date: date | None = None
    storage: str | None = None
    udi: str | None = None

    #: 产品属性：是否无菌提供 / 是否一次性使用（决定条件项是否适用）
    is_sterile: bool | None = None
    is_single_use: bool | None = None

    #: 标签上的符号是否存在（被检查对象）
    sterile_mark: bool | None = None
    single_use_mark: bool | None = None

    warning: str | None = None
    ifu_ref: str | None = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    """整体抽取置信度。低于阈值时，缺失字段一律判「不可判定」而非「违规」。"""

    quality_flags: list[str] = Field(default_factory=list)
    """模糊 / 过曝 / 非标签内容等。非空同样触发「不可判定」。"""

    raw_text: str = ""

    parse_issues: list[str] = Field(default_factory=list)
    """「标注了但解析不出合法值」的字段（存 RequirementKey 的 value）。

    与「未标注」必须区分：前者是格式问题，后者是缺项问题，
    给企业整改的方向完全不同。用错理由会误导对方。
    """

    def get(self, key: RequirementKey):
        return getattr(self, _FIELD_OF[key], None)


class ChecklistItemResult(BaseModel):
    key: RequirementKey
    label: str
    required: bool
    status: ItemStatus
    reason: str
    clause: str
    """依据的法规条款。报告里必须逐项指出来源。"""


class LabelReviewResult(BaseModel):
    risk_level: RiskLevel
    items: list[ChecklistItemResult] = Field(default_factory=list)

    compliant: int = 0
    violations: int = 0
    undeterminable: int = 0

    review_required: bool = True
    """永远是 True：AI 结论是草稿，必须审核员确认后才能出正式报告。"""

    caveats: list[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def undeterminable_rate(self) -> float:
        return self.undeterminable / self.total if self.total else 0.0

    def violation_keys(self) -> list[RequirementKey]:
        return [i.key for i in self.items if i.status is ItemStatus.VIOLATION]


#: RequirementKey → LabelExtraction 字段名。放在这里而不是散在引擎里，
#: 这样加字段只改一处。
_FIELD_OF: dict[RequirementKey, str] = {
    RequirementKey.LABEL_PRODUCT_NAME: "product_name",
    RequirementKey.LABEL_MODEL_SPEC: "model_spec",
    RequirementKey.LABEL_REGISTRANT_NAME: "registrant_name",
    RequirementKey.LABEL_REGISTRANT_ADDRESS: "registrant_address",
    RequirementKey.LABEL_REGISTRANT_CONTACT: "registrant_contact",
    RequirementKey.LABEL_CERT_NO: "cert_no",
    RequirementKey.LABEL_PRODUCTION_LICENSE: "production_license",
    RequirementKey.LABEL_MANUFACTURE_DATE: "manufacture_date",
    RequirementKey.LABEL_EXPIRY_DATE: "expiry_date",
    RequirementKey.LABEL_STORAGE: "storage",
    RequirementKey.LABEL_UDI: "udi",
    RequirementKey.LABEL_STERILE_MARK: "sterile_mark",
    RequirementKey.LABEL_SINGLE_USE: "single_use_mark",
    RequirementKey.LABEL_WARNING: "warning",
    RequirementKey.LABEL_IFU_REF: "ifu_ref",
}


__all__ = [
    "ChecklistItemResult",
    "ItemStatus",
    "LabelExtraction",
    "LabelReviewResult",
    "RiskLevel",
]
