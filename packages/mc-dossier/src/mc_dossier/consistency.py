"""卷宗一致性引擎。

## 它解决的不是检索问题

RAG 能回答"技术要求里写了什么"，但回答不了
「技术要求写 Ø3.20、检验报告写 Ø3.50，谁对？」
—— 那必须**先把多份文档抽成结构化字段，再做对齐与比对**。

## 三态结论

每个属性只有三种结果，缺一不可：
- **冲突**：≥2 份文档都写了，归一化后不相等；
- **一致**：≥2 份文档都写了且归一化后相等；
- **无法核对**：只有 1 份文档提到它 —— 不能算一致（没有比对），也不能算冲突。

第三态是刻意保留的。把它悄悄算进"一致"会让报告虚高，
是这类工具最容易犯的自欺。

## 缺失文件

按 121 号通告的 RPS 章节，一套申报资料应包含的章节是可枚举的，
所以"缺项"可以确定性检出。
"""

from __future__ import annotations

from mc_dossier.normalize import normalize
from mc_dossier.schemas import (
    ATTRIBUTE_LABELS,
    DOC_LABELS,
    Attribute,
    AttributeObservation,
    DocKind,
    Document,
    DossierReviewResult,
    Finding,
    Severity,
)

#: 一套申报资料应当包含的章节（简化版 RPS 目录）。
REQUIRED_DOCS: tuple[DocKind, ...] = (
    DocKind.TECHNICAL_REQUIREMENT,
    DocKind.IFU,
    DocKind.TEST_REPORT,
)

#: 每个属性的严重度。尺寸与型号判错会直接导致发补，地址写错通常只是形式问题。
SEVERITY: dict[Attribute, Severity] = {
    Attribute.DIMENSION: Severity.HIGH,
    Attribute.MODEL_SPEC: Severity.HIGH,
    Attribute.STERILIZATION: Severity.HIGH,
    Attribute.SHELF_LIFE: Severity.MEDIUM,
    Attribute.STRUCTURE: Severity.MEDIUM,
    Attribute.MANUFACTURER_ADDRESS: Severity.LOW,
}

CLAUSE = "《医疗器械注册申报资料要求》一致性要求"


def group_by_attribute(docs: list[Document]) -> dict[Attribute, list[AttributeObservation]]:
    """把各文档写的同一属性聚到一起。归一化只有 1 份时不参与冲突判定。"""
    grouped: dict[Attribute, list[AttributeObservation]] = {}
    for attribute in Attribute:
        observations: list[AttributeObservation] = []
        for doc in docs:
            raw = doc.get(attribute)
            if raw is None:
                continue
            observations.append(
                AttributeObservation(
                    doc_id=doc.doc_id,
                    doc_label=DOC_LABELS[doc.kind],
                    raw_value=raw,
                    normalized_value=normalize(attribute, raw),
                )
            )
        if observations:
            grouped[attribute] = observations
    return grouped


def review_dossier(docs: list[Document], *, dossier_id: str = "dossier") -> DossierReviewResult:
    if not docs:
        raise ValueError("卷宗为空：至少需要一份文档")

    grouped = group_by_attribute(docs)
    findings: list[Finding] = []
    consistent: list[Attribute] = []
    unverifiable: list[Attribute] = []

    for attribute, observations in grouped.items():
        # 有任一侧归一化不出可比较的值（如尺寸缺单位、有效期只写"长期"），
        # 整个属性降级为"无法核对"—— 拿不可比的值去判冲突就是制造假冲突。
        if any(o.normalized_value is None for o in observations):
            unverifiable.append(attribute)
            continue

        distinct = {o.normalized_value for o in observations}
        if len(observations) < 2:
            # 只有一份文档写了它 —— 无从比对。不并入"一致"。
            unverifiable.append(attribute)
            continue
        if len(distinct) == 1:
            consistent.append(attribute)
            continue

        findings.append(
            Finding(
                attribute=attribute,
                label=ATTRIBUTE_LABELS[attribute],
                severity=SEVERITY[attribute],
                message=(
                    f"{ATTRIBUTE_LABELS[attribute]}在不同文件中不一致："
                    + "；".join(f"{o.doc_label}「{o.raw_value}」" for o in observations)
                ),
                observations=observations,
                clause=CLAUSE,
            )
        )

    present = {d.kind for d in docs}
    missing = [k for k in REQUIRED_DOCS if k not in present]

    caveats = [
        "本报告由 AI 自动比对生成，须经审核员复核确认后方可对外出具。",
        "比对范围限于结构化属性（型号、尺寸、灭菌方式、有效期、生产地址、结构组成），"
        "不涵盖技术要求全文与临床评价的实质内容。",
    ]
    if unverifiable:
        caveats.append(
            "以下属性只有一份文件提及，无法交叉核对（不计入一致率）："
            + "、".join(ATTRIBUTE_LABELS[a] for a in unverifiable)
        )
    if missing:
        caveats.append(
            "缺失章节：" + "、".join(DOC_LABELS[k] for k in missing) + "，请补齐后再审查。"
        )

    return DossierReviewResult(
        dossier_id=dossier_id,
        findings=findings,
        missing_docs=missing,
        conflicts=len(findings),
        consistent_attributes=consistent,
        unverifiable=unverifiable,
        caveats=caveats,
    )


def extract_fields_from_text(kind: DocKind, text: str) -> dict[str, str]:
    """从纯文本里抽属性。正则实现，覆盖"标签式"写法。

    真实卷宗是 Word/PDF，抽取应接版面解析；这里保留一个可替换的正则基线，
    让评测能在可控输入上跑出确定结论。
    """
    import re

    patterns: dict[Attribute, tuple[str, ...]] = {
        Attribute.MODEL_SPEC: ("型号规格", "规格型号", "型号"),
        Attribute.DIMENSION: ("关键尺寸", "尺寸", "外径", "内径"),
        Attribute.STERILIZATION: ("灭菌方式", "灭菌方法"),
        Attribute.SHELF_LIFE: ("有效期", "使用期限", "贮存期限"),
        Attribute.MANUFACTURER_ADDRESS: ("生产地址", "住所"),
        Attribute.STRUCTURE: ("结构组成", "结构及组成", "组成"),
    }
    out: dict[str, str] = {}
    for attribute, names in patterns.items():
        for name in names:
            m = re.search(rf"{name}[：:\s]*([^\n；;，,]+)", text)
            if m:
                value = m.group(1).strip(" 　。.")
                if value:
                    out[attribute.value] = value
                    break
    return out


__all__ = [
    "CLAUSE",
    "REQUIRED_DOCS",
    "SEVERITY",
    "extract_fields_from_text",
    "group_by_attribute",
    "review_dossier",
]
