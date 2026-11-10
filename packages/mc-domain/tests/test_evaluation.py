"""评测层的单测：标签归一化、指标、以及闸门的顺序。

这一层出错不会报错，只会让报告里的数字悄悄失真——所以每条规则都要有测试钉住。
"""

from __future__ import annotations

import asyncio

import pytest

from mc_domain.classify import ClassificationPipeline
from mc_domain.enums import ClassificationOutcome
from mc_domain.evaluation import (
    ExpectedOutcome,
    LabelDistribution,
    class_metrics,
    gate_metrics,
    normalize_label,
    wilson_interval,
)
from mc_domain.gate import DeviceGate
from mc_domain.retrieval import KeywordCodeRetriever
from mc_domain.schemas import ProductProfile
from mc_domain.tree import ClassificationTree, CodeNode


# ---------------------------------------------------------------------------
# 标签归一化：官方原文的写法极不统一，这里逐种钉死
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ⅲ", ExpectedOutcome.CLASS_III),
        ("Ⅱ", ExpectedOutcome.CLASS_II),
        ("Ⅰ", ExpectedOutcome.CLASS_I),
        ("第三类", ExpectedOutcome.CLASS_III),
        ("按第二类管理", ExpectedOutcome.CLASS_II),
        ("第一类", ExpectedOutcome.CLASS_I),
        ("不作为医疗器械", ExpectedOutcome.NOT_A_DEVICE),
        ("不属于医疗器械", ExpectedOutcome.NOT_A_DEVICE),
        ("建议不按照医疗器械管理", ExpectedOutcome.NOT_A_DEVICE),
        ("不单独作为医疗器械", ExpectedOutcome.NOT_ALONE),
        ("建议不单独申报", ExpectedOutcome.NOT_ALONE),
        ("药械组合", ExpectedOutcome.COMBINATION),
        ("视具体情况而定", ExpectedOutcome.DEPENDS),
        ("", ExpectedOutcome.OTHER),
        ("莫名其妙", ExpectedOutcome.OTHER),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) is expected


def test_not_alone_is_not_swallowed_by_not_device():
    """「不单独作为医疗器械」里含「作为医疗器械」，
    若先判 not_device 会得到错误结论——顺序错了指标就全歪。"""
    assert normalize_label("不单独作为医疗器械") is ExpectedOutcome.NOT_ALONE
    assert normalize_label("不单独作为医疗器械").is_device is None


def test_class_iii_not_misread_as_class_ii():
    """非法全角罗马数字时 III 会命中 II。这条防的是顺序回归。"""
    assert normalize_label("Ⅲ类") is ExpectedOutcome.CLASS_III
    assert normalize_label("III") is ExpectedOutcome.CLASS_III


def test_scope_predicates():
    assert ExpectedOutcome.CLASS_II.in_class_scope
    assert ExpectedOutcome.CLASS_II.in_gate_scope
    assert ExpectedOutcome.NOT_A_DEVICE.in_gate_scope
    assert not ExpectedOutcome.NOT_A_DEVICE.in_class_scope
    # 系统表达不了的，两类指标都不参与
    for outcome in (
        ExpectedOutcome.NOT_ALONE,
        ExpectedOutcome.COMBINATION,
        ExpectedOutcome.DEPENDS,
        ExpectedOutcome.OTHER,
    ):
        assert not outcome.in_gate_scope
        assert not outcome.in_class_scope
        assert outcome.system_outcome is None
        assert outcome.is_device is None


def test_distribution_counts_unsupported_separately():
    dist = LabelDistribution(total=4)
    for raw in ("Ⅲ", "不作为医疗器械", "药械组合", "视具体情况而定"):
        dist.counts[normalize_label(raw)] += 1
    assert dist.gate_scope == 2
    assert dist.class_scope == 1
    assert dist.unsupported == 2


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def test_wilson_interval_shrinks_with_n():
    lo_small, hi_small = wilson_interval(8, 10)
    lo_big, hi_big = wilson_interval(800, 1000)
    assert (hi_small - lo_small) > (hi_big - lo_big)
    assert hi_small <= 1.0 and lo_big >= 0.0


def test_gate_metrics_counts_default_to_device_cost():
    """全判为器械：召回 100%、特异度 0——这正是规则闸门的真实处境。"""
    pairs = [(True, True), (True, True), (False, True), (False, True)]
    m = gate_metrics(pairs)
    assert m.recall == 1.0
    assert m.specificity == 0.0
    assert m.accuracy == 0.5


def test_gate_metrics_skips_none():
    assert gate_metrics([(True, None), (None, True)]).n == 0


def test_class_metrics_only_counts_given_predictions():
    samples = [
        (ExpectedOutcome.CLASS_II, [ClassificationOutcome.CLASS_II]),
        (ExpectedOutcome.CLASS_II, []),  # 拒答：调用方本不该把它塞进来
    ]
    m = class_metrics(samples)
    assert m.n == 2
    assert m.top1_accuracy == 0.5


# ---------------------------------------------------------------------------
# 闸门
# ---------------------------------------------------------------------------


def test_gate_fires_on_explicit_negative():
    gate = DeviceGate()
    result = gate.evaluate(ProductProfile(intended_use="仅用于清洁，无治疗作用", raw_text="清洁器"))
    assert result.decision.verdict == "not_device"
    assert result.short_circuit is ClassificationOutcome.NOT_A_DEVICE
    assert result.decision.matched == "无治疗作用"


def test_gate_stays_uncertain_without_marker():
    gate = DeviceGate()
    result = gate.evaluate(ProductProfile(intended_use="用于缓解肌肉疼痛", raw_text="治疗仪"))
    assert result.decision.verdict == "uncertain"
    assert result.short_circuit is None


# ---------------------------------------------------------------------------
# 顺序回归：闸门必须早于槽位检查
# ---------------------------------------------------------------------------


class _ExplodingNormalizer:
    async def normalize(self, raw_text: str) -> ProductProfile:  # pragma: no cover
        raise AssertionError("传入 profile 时不应调用 normalizer")


@pytest.fixture
def tiny_tree():
    return ClassificationTree.from_nodes(
        [CodeNode(code="09", name="物理治疗器械", level=0, sort_order=1)]
    )


def test_not_device_short_circuits_before_slot_check(tiny_tree):
    """槽位一个都没有，但闸门判定不是器械 → 应直接给结论，而不是追问接触类型。

    这是评测里真实抓到过的顺序 bug：闸门曾在槽位检查之后，
    导致 264/272 条样本在到达闸门前就被转成追问，门控指标完全测不出来。
    """
    pipeline = ClassificationPipeline(
        tree=tiny_tree,
        retriever=KeywordCodeRetriever(tiny_tree.search_corpus()),
        normalizer=_ExplodingNormalizer(),
    )
    profile = ProductProfile(
        intended_use="仅用于物理遮光，无治疗作用",
        raw_text="儿童遮光眼罩",
        contact_type=None,
        energy_type=None,
    )
    result = asyncio.run(pipeline.classify("儿童遮光眼罩", profile=profile))

    assert result.status == "classified"
    assert result.outcome is ClassificationOutcome.NOT_A_DEVICE
    assert result.best is None
    assert [s.stage for s in result.steps] == ["normalize", "gate"]
