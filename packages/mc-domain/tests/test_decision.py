"""风险偏好与最终取舍策略的单测。

这一段很容易被当成"调参"，所以测试要钉住的不是某个数字，
而是三条不变量：
1. alpha=0 时行为与"取最高分"完全一致（不悄悄改变默认行为）；
2. 只有落在带宽内才会往上跳类别（不会无依据地把 Ⅱ 类判成 Ⅲ 类）；
3. 无论怎么选，结果都必须能说出理由（理由进依据链）。
"""

from __future__ import annotations

import pytest

from mc_domain.decision import (
    DEFAULT_PRESET,
    PRESETS,
    RiskPreference,
    available_presets,
    get_preference,
    select_candidate,
)
from mc_domain.enums import ManagementClass
from mc_domain.schemas import Candidate


def _c(code: str, cls: ManagementClass | None, score: float) -> Candidate:
    return Candidate(code=code, name=code, name_path=code, management_class=cls, score=score)


def test_alpha_zero_matches_top_score():
    cands = [_c("a", ManagementClass.CLASS_II, 10.0), _c("b", ManagementClass.CLASS_III, 9.9)]
    chosen, reason = select_candidate(cands, PRESETS["strict"])
    assert chosen is not None and chosen.code == "a"
    assert "最高分" in reason


def test_balanced_lifts_within_band():
    """9.8 落在 10.0 的 5% 带宽内 → 按较高类别判。"""
    cands = [_c("a", ManagementClass.CLASS_II, 10.0), _c("b", ManagementClass.CLASS_III, 9.8)]
    chosen, reason = select_candidate(cands, PRESETS["balanced"])
    assert chosen is not None and chosen.code == "b"
    assert "就高不就低" in reason


def test_outside_band_does_not_lift():
    """9.0 在 10.0 的 5% 带宽外 → 不得跳类。这是防止无依据拔高风险。"""
    cands = [_c("a", ManagementClass.CLASS_II, 10.0), _c("b", ManagementClass.CLASS_III, 9.0)]
    chosen, _ = select_candidate(cands, PRESETS["balanced"])
    assert chosen is not None and chosen.code == "a"


def test_wider_alpha_lifts_more():
    cands = [_c("a", ManagementClass.CLASS_II, 10.0), _c("b", ManagementClass.CLASS_III, 8.5)]
    assert select_candidate(cands, PRESETS["balanced"])[0].code == "a"
    assert select_candidate(cands, PRESETS["max_recall_iii"])[0].code == "b"


def test_candidates_without_class_are_ignored_for_lifting():
    cands = [
        _c("x", None, 10.0),
        _c("a", ManagementClass.CLASS_I, 10.0),
        _c("b", ManagementClass.CLASS_III, 9.9),
    ]
    chosen, _ = select_candidate(cands, PRESETS["risk_averse"])
    assert chosen is not None and chosen.code == "b"


def test_empty_candidates():
    chosen, reason = select_candidate([], PRESETS["balanced"])
    assert chosen is None
    assert reason


def test_every_preset_returns_a_reason():
    cands = [_c("a", ManagementClass.CLASS_II, 1.0), _c("b", ManagementClass.CLASS_III, 0.9)]
    for name in PRESETS:
        chosen, reason = select_candidate(cands, PRESETS[name])
        assert chosen is not None
        assert reason.strip()


def test_default_preset_exists():
    assert DEFAULT_PRESET in PRESETS
    assert get_preference(None) is PRESETS[DEFAULT_PRESET]


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="未知的风险偏好"):
        get_preference("不存在的策略")


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        RiskPreference(alpha=1.5, label="x", description="x")


def test_presets_are_listed_for_api():
    listing = available_presets()
    assert {p["name"] for p in listing} == set(PRESETS)
    for item in listing:
        assert item["description"]
