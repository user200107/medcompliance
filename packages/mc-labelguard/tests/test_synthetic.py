"""合成标签集与指标的单测。

这层的价值全在**可复现**：同一 seed 必须产出同一批样本，
否则指标会随运行漂移，报告就失去意义。
"""

from __future__ import annotations

from mc_domain.enums import RequirementKey
from mc_labelguard.checklist import check_label
from mc_labelguard.synthetic import DEFECT_RENDERERS, EXPECTED, generate, to_text


def test_generation_is_reproducible():
    a = generate(20, seed=1)
    b = generate(20, seed=1)
    assert [x.text for x in a] == [x.text for x in b]
    assert [x.planted for x in a] == [x.planted for x in b]


def test_different_seed_differs():
    a = generate(30, seed=1)
    b = generate(30, seed=2)
    assert [x.text for x in a] != [x.text for x in b]


def test_every_defect_kind_appears():
    samples = generate(120)
    kinds = {s.defect_kind for s in samples}
    assert kinds == set(DEFECT_RENDERERS)


def test_clean_samples_have_no_planted_defects():
    samples = generate(60)
    clean = [s for s in samples if s.clean]
    assert clean, "应当包含负控样本"
    assert all(s.planted == () for s in clean)


def test_planted_matches_expected_mapping():
    for s in generate(120):
        assert s.planted == EXPECTED[s.defect_kind]


def test_fields_mode_detects_all_planted():
    """字段模式下引擎应当 100% 检出埋入项——这是判定引擎本身的基准。"""
    for s in generate(120):
        result = check_label(s.fields)
        detected = set(result.violation_keys())
        missing = set(s.planted) - detected
        assert not missing, f"{s.defect_kind} 漏检 {missing}"


def test_clean_samples_produce_no_violation():
    for s in generate(60):
        if not s.clean:
            continue
        assert check_label(s.fields).violations == 0, f"{s.label_id} 假阳性"


def test_text_mode_is_blind_to_symbol_presence():
    """文本模式**结构性地**看不到"符号是否存在"这类缺陷。

    在文本里「无菌」这个词既是产品属性也是符号存在，二者无法区分。
    所以这两类缺陷只有视觉抽取才抓得到 —— 这正是接视觉层的量化理由。
    """
    symbol_defects = {"sterile_missing_symbol", "single_use_missing_symbol"}
    for s in generate(120):
        if s.defect_kind not in symbol_defects:
            continue
        from mc_labelguard.extract import extract_from_text

        detected = set(check_label(extract_from_text(s.text)).violation_keys())
        assert not (set(s.planted) & detected), "文本模式不应声称检出了符号缺失"


def test_to_text_omits_missing_fields():
    samples = {s.defect_kind: s for s in generate(60)}
    missing = samples["missing_cert_no"]
    assert "注册证编号" not in to_text(missing.fields)
