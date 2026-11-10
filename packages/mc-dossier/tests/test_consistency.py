"""P3 卷宗一致性引擎的单测。

要钉住的两件事正好相反，所以必须同时有：
1. **该报的必须报**（尺寸 3.20 vs 3.50）；
2. **不该报的绝不能报**（Ø3.20 mm vs 3.2mm、3年 vs 36个月、A-100 vs a 100）。

只测第 1 条会得到"召回高但全是误报"的引擎，审核员三次之后就不看报告了。
"""

from __future__ import annotations

import pytest

from mc_dossier.consistency import (
    REQUIRED_DOCS,
    extract_fields_from_text,
    group_by_attribute,
    review_dossier,
)
from mc_dossier.normalize import (
    normalize_dimension,
    normalize_model,
    normalize_shelf_life,
    normalize_sterilization,
)
from mc_dossier.schemas import Attribute, DocKind, Document, Severity
from mc_dossier.synthetic import build_dossier, generate


def _doc(kind: DocKind, **fields) -> Document:
    return Document(
        doc_id=kind.value,
        kind=kind,
        filename=f"{kind.value}.docx",
        fields={k: v for k, v in fields.items()},
        text="",
    )


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Ø3.20 mm", "3.2mm"),
        ("3.20MM", "3.2 mm"),
        ("3.2毫米", "3.2mm"),
    ],
)
def test_dimension_rewording_is_equal(a, b):
    assert normalize_dimension(a) == normalize_dimension(b)


def test_dimension_real_difference_survives():
    assert normalize_dimension("Ø3.20 mm") != normalize_dimension("Ø3.50 mm")


def test_centimeter_converts():
    assert normalize_dimension("1.2cm") == normalize_dimension("12mm")


def test_model_rewording_is_equal():
    assert normalize_model("A-100") == normalize_model("a 100")


def test_model_difference_survives():
    assert normalize_model("A-100") != normalize_model("A-101")


def test_shelf_life_years_vs_months():
    assert normalize_shelf_life("3年") == normalize_shelf_life("36个月")
    assert normalize_shelf_life("3年") != normalize_shelf_life("24个月")


def test_sterilization_synonyms():
    assert normalize_sterilization("环氧乙烷灭菌") == normalize_sterilization("EO")
    assert normalize_sterilization("环氧乙烷") != normalize_sterilization("辐照灭菌")


# ---------------------------------------------------------------------------
# 引擎：负控（写法不同 → 不许报）
# ---------------------------------------------------------------------------


def test_base_dossier_has_no_conflicts():
    """基线卷宗里的写法差异（3.20mm/3.2mm、3年/36个月、A-100/a 100）都不许报冲突。"""
    dossier = build_dossier("clean", dossier_id="d0")
    result = review_dossier(dossier.docs, dossier_id=dossier.dossier_id)
    assert result.conflicts == 0
    assert Attribute.DIMENSION in result.consistent_attributes
    assert Attribute.SHELF_LIFE in result.consistent_attributes


def test_rewording_only_produces_no_conflict():
    dossier = build_dossier("rewording_only", dossier_id="d1")
    result = review_dossier(dossier.docs)
    assert result.conflicts == 0


# ---------------------------------------------------------------------------
# 引擎：真冲突（必须报）
# ---------------------------------------------------------------------------


def test_dimension_conflict_detected_with_both_values():
    dossier = build_dossier("dimension_mismatch", dossier_id="d2")
    result = review_dossier(dossier.docs)

    assert result.conflicts == 1
    finding = result.findings[0]
    assert finding.attribute is Attribute.DIMENSION
    assert finding.severity is Severity.HIGH
    # 报告必须同时列出两边的原值，否则审核员无法判断谁对
    values = {o.raw_value for o in finding.observations}
    assert "Ø3.20 mm" in values and "Ø3.50 mm" in values


def test_model_conflict_is_high_severity():
    result = review_dossier(build_dossier("model_mismatch", dossier_id="d3").docs)
    assert result.high_severity == 1
    assert result.findings[0].attribute is Attribute.MODEL_SPEC


def test_low_severity_attribute():
    result = review_dossier(build_dossier("address_mismatch", dossier_id="d4").docs)
    assert result.findings[0].severity is Severity.LOW


def test_missing_doc_detected():
    dossier = build_dossier("missing_test_report", dossier_id="d5", drop_doc=DocKind.TEST_REPORT)
    result = review_dossier(dossier.docs)
    assert DocKind.TEST_REPORT in result.missing_docs
    assert any("缺失章节" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# 第三态：无法核对
# ---------------------------------------------------------------------------


def test_single_mention_is_unverifiable_not_consistent():
    """只有一份文档提到该属性时，既不能算一致（没比对），也不能算冲突。"""
    docs = [_doc(DocKind.TECHNICAL_REQUIREMENT, model_spec="A-100")]
    result = review_dossier(docs)
    assert Attribute.MODEL_SPEC in result.unverifiable
    assert Attribute.MODEL_SPEC not in result.consistent_attributes


def test_group_by_attribute_collects_all_mentions():
    dossier = build_dossier("dimension_mismatch", dossier_id="d6")
    grouped = group_by_attribute(dossier.docs)
    assert len(grouped[Attribute.DIMENSION]) == 2


# ---------------------------------------------------------------------------
# 文本抽取（可替换基线）
# ---------------------------------------------------------------------------


def test_extract_fields_from_text():
    text = "型号规格：A-100\n关键尺寸：Ø3.20 mm\n灭菌方式：环氧乙烷灭菌\n"
    fields = extract_fields_from_text(DocKind.TECHNICAL_REQUIREMENT, text)
    assert fields["model_spec"] == "A-100"
    assert fields["dimension"] == "Ø3.20 mm"


def test_empty_dossier_rejected():
    with pytest.raises(ValueError, match="卷宗为空"):
        review_dossier([])


# ---------------------------------------------------------------------------
# 合成集
# ---------------------------------------------------------------------------


def test_generation_is_reproducible():
    a = generate(30, seed=7)
    b = generate(30, seed=7)
    assert [d.kind for d in a] == [d.kind for d in b]
    assert [len(d.docs) for d in a] == [len(d.docs) for d in b]


def test_required_docs_constant_sane():
    assert set(REQUIRED_DOCS) == {
        DocKind.TECHNICAL_REQUIREMENT,
        DocKind.IFU,
        DocKind.TEST_REPORT,
    }
