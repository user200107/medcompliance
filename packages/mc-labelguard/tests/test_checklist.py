"""必标项引擎的单测。

要钉住的核心不变量：
1. **低置信度 / 图像质量问题 → 缺失项只能是 `undeterminable`，不能是 `violation`。**
   这是整个 P2 最重要的一条：把"看不清"判成"违规"是在冤枉企业，
   判成"合规"是在放过风险。
2. 条件项不满足时判 compliant 且写明"不适用"，而不是判缺项。
3. 每条判定都带条款。
"""

from __future__ import annotations

from datetime import date

import pytest

from mc_domain.enums import RequirementKey
from mc_labelguard.checklist import RULES, check_label
from mc_labelguard.extract import extract_from_text
from mc_labelguard.schemas import ItemStatus, LabelExtraction, RiskLevel


def _good(**overrides) -> LabelExtraction:
    base = dict(
        product_name="一次性使用无菌注射器",
        model_spec="2ml",
        registrant_name="某医疗器械有限公司",
        registrant_address="江苏省苏州市某路 1 号",
        registrant_contact="0512-12345678",
        cert_no="国械注准20243010123",
        production_license="苏食药监械生产许20010001号",
        manufacture_date=date(2026, 1, 1),
        expiry_date=date(2029, 1, 1),
        storage="常温干燥处保存",
        udi="(01)06931234567890",
        is_sterile=True,
        is_single_use=True,
        sterile_mark=True,
        single_use_mark=True,
        warning="一次性使用，用后销毁",
        ifu_ref="使用说明书",
        confidence=0.9,
    )
    base.update(overrides)
    return LabelExtraction(**base)


def test_complete_label_is_low_risk():
    result = check_label(_good())
    assert result.violations == 0
    assert result.risk_level is RiskLevel.LOW
    assert result.review_required is True


def test_missing_required_item_blocks_when_confident():
    result = check_label(_good(cert_no=None))
    assert result.risk_level is RiskLevel.HIGH
    assert RequirementKey.LABEL_CERT_NO in result.violation_keys()


def test_low_confidence_never_produces_violation():
    """置信度不足时，缺失必须是「不可判定」。"""
    result = check_label(_good(cert_no=None, confidence=0.3))
    assert result.violations == 0
    assert RequirementKey.LABEL_CERT_NO not in result.violation_keys()
    statuses = {i.key: i.status for i in result.items}
    assert statuses[RequirementKey.LABEL_CERT_NO] is ItemStatus.UNDETERMINABLE


def test_quality_flags_block_judgement():
    result = check_label(_good(cert_no=None, quality_flags=["图片模糊"]))
    assert result.violations == 0
    assert any("图片模糊" in c for c in result.caveats)


def test_bad_cert_format_is_violation_even_with_high_confidence():
    """格式错误是**确定**的违规，不受置信度影响——因为字段本身读出来了。"""
    result = check_label(_good(cert_no="ABC-123"))
    assert RequirementKey.LABEL_CERT_NO in result.violation_keys()


def test_conditional_item_not_applicable_is_compliant():
    """非无菌产品不要求无菌标识，且理由要写明「不适用」。"""
    result = check_label(_good(is_sterile=False, sterile_mark=None))
    item = next(i for i in result.items if i.key is RequirementKey.LABEL_STERILE_MARK)
    assert item.status is ItemStatus.COMPLIANT
    assert "不适用" in item.reason


def test_sterile_product_missing_symbol_is_violation():
    """真正的漏洞场景：产品是无菌的，但标签上没标无菌符号 —— 必须判违规。

    这正是把「产品属性」与「符号存在」混成一个字段会漏掉的那类问题：
    字段只要有值（无论 True/False）就会被当成"已标注"。
    """
    result = check_label(_good(is_sterile=True, sterile_mark=None))
    assert RequirementKey.LABEL_STERILE_MARK in result.violation_keys()


def test_every_item_carries_clause():
    result = check_label(_good())
    assert all(i.clause for i in result.items)
    assert len(result.items) >= len(RULES)


def test_optional_item_missing_is_not_violation():
    """非强制项缺失不是违规。判成违规会制造假阳性，
    让企业去整改一个法规并不要求它标注的项。"""
    result = check_label(_good(warning=None, ifu_ref=None, production_license=None))
    assert result.violations == 0
    for key in (
        RequirementKey.LABEL_WARNING,
        RequirementKey.LABEL_IFU_REF,
        RequirementKey.LABEL_PRODUCTION_LICENSE,
    ):
        item = next(i for i in result.items if i.key is key)
        assert item.status is ItemStatus.COMPLIANT
        assert "非强制项" in item.reason


def test_date_order_violation_detected():
    result = check_label(_good(manufacture_date=date(2030, 1, 1), expiry_date=date(2029, 1, 1)))
    assert any("生产日期" in i.label for i in result.items if i.status is ItemStatus.VIOLATION)


def test_undeterminable_rate_reported():
    result = check_label(_good(cert_no=None, udi=None, confidence=0.2))
    assert result.undeterminable_rate > 0
    assert any("不可判定" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# 文本抽取
# ---------------------------------------------------------------------------


SAMPLE_LABEL = """产品名称：一次性使用无菌注射器
型号规格：2ml
注册人名称：某医疗器械有限公司
注册人住所：江苏省苏州市某路 1 号
联系方式：0512-12345678
医疗器械注册证编号：国械注准20243010123
生产许可证编号：苏食药监械生产许20010001号
生产日期：2026-01-01
失效日期：2029-01-01
贮存条件：常温干燥处保存
UDI：(01)06931234567890
无菌
一次性使用
注意事项：一次性使用，用后销毁
使用说明书
"""


def test_extract_from_text_full_label():
    e = extract_from_text(SAMPLE_LABEL)
    assert e.product_name == "一次性使用无菌注射器"
    assert e.cert_no == "国械注准20243010123"
    assert e.udi == "(01)06931234567890"
    assert e.manufacture_date == date(2026, 1, 1)
    assert e.expiry_date == date(2029, 1, 1)
    assert e.sterile_mark is True
    assert e.single_use_mark is True
    assert e.confidence >= 0.6  # 文本模式视为可靠读取，缺失才判违规


def test_extract_respects_negation():
    """「非无菌」「可重复使用」不能被正向关键词吃掉。"""
    e = extract_from_text("产品名称：手术器械\n非无菌提供\n可重复使用\n")
    assert e.sterile_mark is False
    assert e.single_use_mark is False


def test_extract_then_check_flags_missing_items():
    text = SAMPLE_LABEL.replace("医疗器械注册证编号：国械注准20243010123\n", "")
    result = check_label(extract_from_text(text))
    assert RequirementKey.LABEL_CERT_NO in result.violation_keys()
    assert result.risk_level is RiskLevel.HIGH


def test_extract_then_check_passes_complete_label():
    result = check_label(extract_from_text(SAMPLE_LABEL))
    assert result.violations == 0
    assert result.risk_level is RiskLevel.LOW
