"""分类判定流水线的单测。

不需要数据库、不需要向量库、不调模型——全部是纯函数与假实现。
这是刻意的：分类规则会反复调整，如果每改一次都要起全套中间件，最后就没人跑了。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mc_domain import (
    ClassificationPipeline,
    KeywordCodeRetriever,
    ProductProfile,
    build_tree,
    extract_exclusion_terms,
    load_csv,
)
from mc_domain.classify import _apply_most_specific
from mc_domain.enums import ContactType, EnergyType, ManagementClass
from mc_domain.tree import ClassificationTree, CodeNode
from mc_domain.schemas import Candidate

DATA_CSV = (
    Path(__file__).resolve().parents[1] / "src" / "mc_domain" / "data" / "catalog_sample.csv"
)


class ExplodingNormalizer:
    """显式传入 profile 时，normalizer 绝不该被调用。

    这条断言很重要：它把「模型不在判定路径上」这个设计主张钉死在测试里。
    """

    async def normalize(self, raw_text: str) -> ProductProfile:  # pragma: no cover
        raise AssertionError("传入 profile 时不应调用 normalizer")


@pytest.fixture(scope="module")
def tree():
    report = load_csv(DATA_CSV, catalog_version="sample-synthetic")
    assert report.ok, report.problems
    return build_tree(report.nodes, "sample-synthetic")


@pytest.fixture
def pipeline(tree):
    retriever = KeywordCodeRetriever(tree.search_corpus())
    return ClassificationPipeline(tree, retriever, ExplodingNormalizer())


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 树
# ---------------------------------------------------------------------------


def test_tree_structure(tree):
    assert len(tree) == 8
    assert [r.code for r in tree.roots()] == ["01", "07", "09"]
    assert tree.validate() == []


def test_tree_paths_and_ancestors(tree):
    leaf = tree.get("07-01-01")
    assert leaf.path == "07/07-01/07-01-01"
    assert leaf.name_path == "医用成像器械/超声诊断设备/眼科超声诊断设备"
    assert [a.code for a in tree.ancestors("07-01-01")] == ["07", "07-01"]
    assert tree.is_within("07-01-01", "07")
    assert not tree.is_within("09-01", "07")


def test_tree_validate_catches_missing_parent():
    report = load_csv(DATA_CSV, catalog_version="v1")
    orphan = report.nodes[0]
    orphan.parent_code = "99"
    with pytest.raises(ValueError, match="父节点"):
        build_tree(report.nodes, "v1")


# ---------------------------------------------------------------------------
# 排除法
# ---------------------------------------------------------------------------


def test_extract_exclusion_terms():
    terms = extract_exclusion_terms("本类不包括：超声治疗设备、超声理疗仪")
    assert terms == ["超声治疗设备", "超声理疗仪"]


def test_exclusion_terms_ignore_short_fragments():
    assert extract_exclusion_terms("本类不包括：其他") == []


def test_exclusion_matches_treatment_device(tree):
    from mc_domain import ExclusionMatcher

    hit = ExclusionMatcher().match(tree.get("07"), "本项目为超声治疗设备，用于缓解疼痛")
    assert hit is not None
    assert hit.matched_term == "超声治疗设备"


# ---------------------------------------------------------------------------
# 端到端：排除法把治疗类产品从成像类里赶出去
# ---------------------------------------------------------------------------


def test_treatment_device_excluded_from_imaging_and_lands_in_0901(pipeline):
    profile = ProductProfile(
        name="超声治疗设备",
        intended_use="用于缓解肌肉疼痛，促进血液循环",
        contact_type=ContactType.SURFACE,
        energy_type=EnergyType.ACTIVE,
    )
    result = run(pipeline.classify("超声治疗设备", profile=profile))

    assert result.status == "classified"
    assert result.best is not None
    assert result.best.code == "09-01"
    assert result.best.management_class is ManagementClass.CLASS_III

    # 被排除的候选必须留痕
    assert any(r.code == "07" for r in result.rejected)
    assert any(s.rule_key == "exclusion" for s in result.steps)


def test_diagnostic_device_lands_under_imaging(pipeline):
    profile = ProductProfile(
        name="眼科超声诊断设备",
        intended_use="用于眼科疾病检查，观察眼底病变",
        contact_type=ContactType.SURFACE,
        energy_type=EnergyType.ACTIVE,
    )
    result = run(pipeline.classify("眼科超声诊断设备", profile=profile))

    assert result.best is not None
    assert result.best.code == "07-01-01"
    assert result.best.management_class is ManagementClass.CLASS_II
    # 每个结论都必须带目录条目依据
    assert any(e.kind == "catalog_entry" for e in result.best.evidence)


def test_missing_advisory_slot_still_gives_conclusion(pipeline):
    """接触类型缺失不再阻断：出结论，但必须标注「未确认维度」。

    这是"降级但可见"的契约。如果悄悄出结论不标注，
    用户会以为结论建立在完整信息上——那比拒绝回答更危险。
    """
    profile = ProductProfile(
        name="眼科超声诊断设备",
        intended_use="用于眼科疾病检查",
        contact_type=None,  # 故意缺失
        energy_type=EnergyType.ACTIVE,
    )
    result = run(pipeline.classify("眼科超声诊断设备", profile=profile))

    assert result.status == "classified"
    assert result.best is not None
    # 只关心接触类型被标记；其余标注性槽位（时长/无菌/复用）同样未确认，不逐一断言
    assert result.unconfirmed[0] == "contact_type"
    assert any("接触类型" in c for c in result.caveats)
    # 仍然提供追问入口，只是不再阻断；每轮最多 2 个
    assert result.missing_slots[0].slot == "contact_type"
    assert 1 <= len(result.missing_slots) <= 2


def test_no_usable_text_still_blocks(pipeline):
    """阻断条件改成「能否拼出检索文本」：一个字都没有时仍然追问。"""
    profile = ProductProfile(name=None, intended_use=None, raw_text="")
    result = run(pipeline.classify("", profile=profile))

    assert result.status == "needs_more_info"
    assert result.best is None
    assert [s.slot for s in result.missing_slots] == ["intended_use"]


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def _cand(code: str, cls: ManagementClass | None, score: float = 1.0) -> Candidate:
    return Candidate(code=code, name=code, name_path=code, management_class=cls, score=score)


def test_most_specific_rule_drops_ancestor():
    kept, dropped = _apply_most_specific([_cand("07-01", ManagementClass.CLASS_II), _cand("07-01-01", ManagementClass.CLASS_II)])
    assert dropped == 1
    assert [c.code for c in kept] == ["07-01-01"]


def test_most_specific_keeps_unrelated():
    kept, dropped = _apply_most_specific(
        [_cand("07-01-01", ManagementClass.CLASS_II), _cand("07-01-02", ManagementClass.CLASS_II)]
    )
    assert dropped == 0
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# 粗筛召回：容器节点的检索文本必须聚合子树（真实评测里 8/53 no_match 的根因）
# ---------------------------------------------------------------------------


def _ivd_tree():
    nodes = [
        CodeNode(code="22", name="临床检验器械", level=0, sort_order=1),
        CodeNode(
            code="22-04",
            name="分子生物学分析设备",
            level=1,
            parent_code="22",
            management_class="III",
            examples=["基因测序用测序试剂盒", "基因测序文库试剂盒"],
        ),
    ]
    return ClassificationTree.from_nodes(nodes)


def test_plain_corpus_fails_coarse_recall_on_thin_container():
    """不开聚合时，子目录文本只有「临床检验器械」8 个字，
    与「基因测序用测序试剂盒」的 bigram 重叠为 0 —— 这就是线上 no_match 的成因。"""
    tree = _ivd_tree()
    retriever = KeywordCodeRetriever(tree.search_corpus())
    assert retriever.search_codes("基因测序用测序试剂盒", limit=5, allowed_codes={"22"}) == []


def test_aggregated_corpus_fixes_coarse_recall():
    tree = _ivd_tree()
    retriever = KeywordCodeRetriever(tree.search_corpus(aggregate_containers=True))
    hits = retriever.search_codes("基因测序用测序试剂盒", limit=5, allowed_codes={"22"})
    assert hits, "聚合后子目录应能被召回"
    assert hits[0][0] == "22"


def test_aggregation_keeps_leaf_text_unchanged():
    """聚合只影响容器节点；叶子节点的检索文本不能被改动，否则会污染排序。"""
    tree = _ivd_tree()
    plain = dict(tree.search_corpus())
    agg = dict(tree.search_corpus(aggregate_containers=True))
    assert plain["22-04"] == agg["22-04"]
    assert plain["22"] != agg["22"]


# ---------------------------------------------------------------------------
# 歧义判定：只在结论层分歧时才追问
# ---------------------------------------------------------------------------


def test_same_class_close_scores_is_not_ambiguous():
    """同类别、不同编码、分数接近 —— 不算歧义，因为注册路径一样。"""
    from mc_domain.classify import _is_ambiguous

    cands = [_cand("07-01-01", ManagementClass.CLASS_II, 10.0),
             _cand("07-01-02", ManagementClass.CLASS_II, 9.9)]
    assert _is_ambiguous(cands) is False


def test_different_class_close_scores_is_ambiguous():
    from mc_domain.classify import _is_ambiguous

    cands = [_cand("07-01-01", ManagementClass.CLASS_II, 10.0),
             _cand("09-01", ManagementClass.CLASS_III, 9.9)]
    assert _is_ambiguous(cands) is True


def test_different_class_clear_margin_is_not_ambiguous():
    from mc_domain.classify import _is_ambiguous

    cands = [_cand("07-01-01", ManagementClass.CLASS_II, 10.0),
             _cand("09-01", ManagementClass.CLASS_III, 5.0)]
    assert _is_ambiguous(cands) is False
