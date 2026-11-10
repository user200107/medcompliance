"""P3 卷宗审查的数据契约。

核心对象是**「同一个属性在几份文件里各写了什么」**，而不是"某份文件说了什么"。
所以这里的建模重心是从 `Document` 抽出 `Attribute`，再按属性分组比对。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DocKind(str, Enum):
    """申报资料的章节类型（对应 121 号通告的 RPS 章节，做了简化）。"""

    TECHNICAL_REQUIREMENT = "technical_requirement"  # 产品技术要求
    IFU = "ifu"  # 说明书
    TEST_REPORT = "test_report"  # 检验报告
    RESEARCH = "research"  # 研究资料
    CLINICAL = "clinical"  # 临床评价资料


DOC_LABELS: dict[DocKind, str] = {
    DocKind.TECHNICAL_REQUIREMENT: "产品技术要求",
    DocKind.IFU: "说明书",
    DocKind.TEST_REPORT: "检验报告",
    DocKind.RESEARCH: "研究资料",
    DocKind.CLINICAL: "临床评价资料",
}


class Attribute(str, Enum):
    """需要跨文档核对的属性。

    刻意只选**有确定值**的属性：型号、尺寸、灭菌方式、有效期、生产地址、结构组成。
    「预期用途」这类自由文本不纳入——它没有唯一正确写法，
    拿它做一致性比对只会产生噪音。
    """

    MODEL_SPEC = "model_spec"
    DIMENSION = "dimension"
    STERILIZATION = "sterilization"
    SHELF_LIFE = "shelf_life"
    MANUFACTURER_ADDRESS = "manufacturer_address"
    STRUCTURE = "structure"


ATTRIBUTE_LABELS: dict[Attribute, str] = {
    Attribute.MODEL_SPEC: "型号规格",
    Attribute.DIMENSION: "关键尺寸",
    Attribute.STERILIZATION: "灭菌方式",
    Attribute.SHELF_LIFE: "有效期/使用期限",
    Attribute.MANUFACTURER_ADDRESS: "生产地址",
    Attribute.STRUCTURE: "结构组成",
}


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Document(BaseModel):
    """一份申报资料。

    `fields` 是已经抽好的属性（键用 `Attribute.value`），
    `text` 保留原文用于给出处。抽字段与比字段分开，
    这样换抽取方式（正则 / LLM）不用改比对逻辑。
    """

    model_config = ConfigDict(protected_namespaces=())

    doc_id: str
    kind: DocKind
    filename: str
    fields: dict[str, str] = Field(default_factory=dict)
    text: str = ""

    def get(self, attribute: Attribute) -> str | None:
        value = self.fields.get(attribute.value)
        return value.strip() if value and value.strip() else None


class AttributeObservation(BaseModel):
    doc_id: str
    doc_label: str
    raw_value: str
    normalized_value: str | None
    """None 表示**归一化不出可比对的值**（如尺寸缺单位）。此时该项应判"无法核对"。"""


class Finding(BaseModel):
    """一条审查发现。

    `observations` 是关键：报告要让审核员一眼看到"技术要求写 3.2、检验报告写 3.5"，
    而不是只给一句"存在冲突"。
    """

    attribute: Attribute
    label: str
    severity: Severity
    message: str
    observations: list[AttributeObservation] = Field(default_factory=list)
    clause: str = ""

    @property
    def is_conflict(self) -> bool:
        return len(self.observations) > 1


class DossierReviewResult(BaseModel):
    dossier_id: str
    findings: list[Finding] = Field(default_factory=list)
    missing_docs: list[DocKind] = Field(default_factory=list)
    conflicts: int = 0
    consistent_attributes: list[Attribute] = Field(default_factory=list)
    unverifiable: list[Attribute] = Field(default_factory=list)
    """只有一份文档写了该属性，无从比对——不能算一致，也不能算冲突。"""

    review_required: bool = True
    caveats: list[str] = Field(default_factory=list)

    @property
    def high_severity(self) -> int:
        return sum(1 for f in self.findings if f.severity is Severity.HIGH)


__all__ = [
    "ATTRIBUTE_LABELS",
    "Attribute",
    "AttributeObservation",
    "DOC_LABELS",
    "DocKind",
    "Document",
    "DossierReviewResult",
    "Finding",
    "Severity",
]
