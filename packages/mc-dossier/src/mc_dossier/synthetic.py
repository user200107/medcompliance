"""合成卷宗：人为埋入已知冲突，用于给 P3 算指标。

## 两类必须同时存在的样本

1. **真冲突**（该报）：尺寸从 3.20 改成 3.50；
2. **仅写法不同**（不该报）：`Ø3.20 mm` vs `3.2mm`、`3年` vs `36个月`、`A-100` vs `a 100`。

只埋第 1 类，会得到一个"召回很高但全是误报"的报告；只埋第 2 类，
会得到一个"精确率很高但什么都没抓到"的报告。**两类必须一起测**——
这是 P3 的精确率有意义的前提。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mc_dossier.consistency import extract_fields_from_text
from mc_dossier.schemas import Attribute, DocKind, Document

#: 基线：三份文档，写法各不相同但归一化后应全部一致（负控）
_BASE: dict[DocKind, dict[Attribute, str]] = {
    DocKind.TECHNICAL_REQUIREMENT: {
        Attribute.MODEL_SPEC: "A-100",
        Attribute.DIMENSION: "Ø3.20 mm",
        Attribute.STERILIZATION: "环氧乙烷灭菌",
        Attribute.SHELF_LIFE: "3年",
        Attribute.STRUCTURE: "由针筒、推杆、活塞、针头组成",
        # 技术要求里通常也有企业/生产地址信息；缺了它，"地址不一致"这类缺陷
        # 会因为只有一份文档提及而被判成"无法核对"，永远测不出来。
        Attribute.MANUFACTURER_ADDRESS: "江苏省苏州市工业园区某路1号",
    },
    DocKind.IFU: {
        Attribute.MODEL_SPEC: "A-100",
        Attribute.SHELF_LIFE: "36个月",
        Attribute.STERILIZATION: "环氧乙烷",
        Attribute.MANUFACTURER_ADDRESS: "江苏省苏州市工业园区某路1号",
    },
    DocKind.TEST_REPORT: {
        Attribute.MODEL_SPEC: "A-100",
        Attribute.DIMENSION: "3.2mm",
        Attribute.STERILIZATION: "EO",
    },
}

_KIND_LABELS = {
    DocKind.TECHNICAL_REQUIREMENT: ("t1", "产品技术要求.docx"),
    DocKind.IFU: ("t2", "说明书.docx"),
    DocKind.TEST_REPORT: ("t3", "检验报告.pdf"),
}

#: 缺陷 → (改哪份文档, 改哪个属性, 改成什么)
DEFECTS: dict[str, tuple[DocKind, Attribute, str]] = {
    "dimension_mismatch": (DocKind.TEST_REPORT, Attribute.DIMENSION, "Ø3.50 mm"),
    "model_mismatch": (DocKind.TEST_REPORT, Attribute.MODEL_SPEC, "A-101"),
    "sterilization_mismatch": (DocKind.TEST_REPORT, Attribute.STERILIZATION, "辐照灭菌"),
    "shelf_life_mismatch": (DocKind.IFU, Attribute.SHELF_LIFE, "24个月"),
    "address_mismatch": (DocKind.IFU, Attribute.MANUFACTURER_ADDRESS, "江苏省南京市某路9号"),
    "structure_mismatch": (
        DocKind.TEST_REPORT,
        Attribute.STRUCTURE,
        "由针筒、推杆、活塞组成",
    ),
}

DEFECT_LABELS: dict[str, str] = {
    "dimension_mismatch": "尺寸不一致（3.20 → 3.50）",
    "model_mismatch": "型号不一致（A-100 → A-101）",
    "sterilization_mismatch": "灭菌方式不一致",
    "shelf_life_mismatch": "有效期不一致（36 → 24 个月）",
    "address_mismatch": "生产地址不一致",
    "structure_mismatch": "结构组成不一致",
    "missing_test_report": "缺少检验报告章节",
    "rewording_only": "仅写法不同（不应报冲突）",
    "hard_rewording": "写法差异更大：尺寸缺单位（应判无法核对）",
    "clean": "无缺陷（负控）",
}

#: 每个缺陷期望检出的属性
EXPECTED_CONFLICTS: dict[str, tuple[Attribute, ...]] = {
    "dimension_mismatch": (Attribute.DIMENSION,),
    "model_mismatch": (Attribute.MODEL_SPEC,),
    "sterilization_mismatch": (Attribute.STERILIZATION,),
    "shelf_life_mismatch": (Attribute.SHELF_LIFE,),
    "address_mismatch": (Attribute.MANUFACTURER_ADDRESS,),
    "structure_mismatch": (Attribute.STRUCTURE,),
    "missing_test_report": (),
    "rewording_only": (),
    "hard_rewording": (),
    "clean": (),
}

#: 仅写法不同的替换（考验归一化，期望零冲突）
_REWORDING: dict[DocKind, dict[Attribute, str]] = {
    DocKind.TEST_REPORT: {
        Attribute.DIMENSION: "3.20MM",
        Attribute.MODEL_SPEC: "a 100",
        Attribute.STERILIZATION: "环氧乙烷(EO)灭菌",
    },
    DocKind.IFU: {
        Attribute.SHELF_LIFE: "36 个月",
    },
}


@dataclass
class SyntheticDossier:
    dossier_id: str
    docs: list[Document]
    planted_conflicts: tuple[Attribute, ...]
    planted_missing: tuple[DocKind, ...]
    kind: str
    expected_conflicts: tuple[Attribute, ...] = field(default=())

    @property
    def clean(self) -> bool:
        return not self.planted_conflicts and not self.planted_missing


def _render_text(kind: DocKind, fields: dict[Attribute, str]) -> str:
    lines = [f"【{_KIND_LABELS[kind][1]}】"]
    label_map = {
        Attribute.MODEL_SPEC: "型号规格",
        Attribute.DIMENSION: "关键尺寸",
        Attribute.STERILIZATION: "灭菌方式",
        Attribute.SHELF_LIFE: "有效期",
        Attribute.MANUFACTURER_ADDRESS: "生产地址",
        Attribute.STRUCTURE: "结构组成",
    }
    for attribute, value in fields.items():
        lines.append(f"{label_map[attribute]}：{value}")
    return "\n".join(lines) + "\n"


def build_dossier(
    kind: str,
    *,
    dossier_id: str,
    drop_doc: DocKind | None = None,
    override: dict[DocKind, dict[Attribute, str]] | None = None,
) -> SyntheticDossier:
    """按缺陷类型构造一份卷宗。

    `kind` 会**自动应用对应缺陷**（`DEFECTS` / `_REWORDING` / 缺章节），
    调用方不需要自己拼 override —— 否则很容易构造出"看着像缺陷、其实没改"的样本，
    指标就白算了（这个坑在单测里被直接抓到过一次）。
    """
    if drop_doc is None and kind == "missing_test_report":
        drop_doc = DocKind.TEST_REPORT

    if override is None:
        if kind in DEFECTS:
            doc_kind, attribute, value = DEFECTS[kind]
            override = {doc_kind: {attribute: value}}
        elif kind == "rewording_only":
            override = _REWORDING
        elif kind == "hard_rewording":
            # 尺寸写成 "3.2"（无单位）：工具无权假定是 mm，
            # 所以该项应判「无法核对」，而不是与 "Ø3.20 mm" 冲突。
            override = {DocKind.TEST_REPORT: {Attribute.DIMENSION: "3.2"}}

    docs: list[Document] = []
    for doc_kind, base_fields in _BASE.items():
        if doc_kind is drop_doc:
            continue
        merged = dict(base_fields)
        merged.update((override or {}).get(doc_kind, {}))
        doc_id, filename = _KIND_LABELS[doc_kind]
        docs.append(
            Document(
                doc_id=doc_id,
                kind=doc_kind,
                filename=filename,
                fields={a.value: v for a, v in merged.items()},
                text=_render_text(doc_kind, merged),
            )
        )

    missing = (drop_doc,) if drop_doc else ()
    return SyntheticDossier(
        dossier_id=dossier_id,
        docs=docs,
        planted_conflicts=EXPECTED_CONFLICTS[kind],
        planted_missing=missing,
        kind=kind,
        expected_conflicts=EXPECTED_CONFLICTS[kind],
    )


def generate(n: int, *, seed: int = 20261110) -> list[SyntheticDossier]:
    """缺陷类型均匀轮转，约 1/3 为负控（clean / rewording_only）。"""
    rng = random.Random(seed)
    kinds = list(DEFECT_LABELS)
    out: list[SyntheticDossier] = []

    for i in range(n):
        kind = kinds[i % len(kinds)]
        dossier_id = f"syn-{i:04d}"

        if kind == "missing_test_report":
            out.append(build_dossier(kind, dossier_id=dossier_id, drop_doc=DocKind.TEST_REPORT))
        elif kind == "rewording_only":
            out.append(build_dossier(kind, dossier_id=dossier_id, override=_REWORDING))
        elif kind in DEFECTS:
            doc_kind, attribute, value = DEFECTS[kind]
            out.append(
                build_dossier(
                    kind,
                    dossier_id=dossier_id,
                    override={doc_kind: {attribute: value}},
                )
            )
        else:  # clean
            # 随机打乱抽一份"写法扰动"，让负控不完全雷同
            jitter = rng.choice([{}, _REWORDING])
            out.append(build_dossier(kind, dossier_id=dossier_id, override=jitter))

    return out


__all__ = [
    "DEFECTS",
    "DEFECT_LABELS",
    "EXPECTED_CONFLICTS",
    "SyntheticDossier",
    "build_dossier",
    "generate",
]
