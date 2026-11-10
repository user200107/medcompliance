"""最终结论的取舍策略（把风险偏好显式化）。

## 为什么这是一个独立模块

分类目录给出的候选往往跨多个管理类别。选谁不是纯技术问题，而是**风险偏好问题**：

- Ⅱ 类走省级注册、Ⅲ 类走国家局审评 —— 判低了，企业会走完整条错流程再被打回；
- Ⅲ 类多为植入/生命支持类 —— 判低了，是把高风险产品当低风险放过去。

所以"取最高分"并不是天然正确的默认值，它只是**不偏向任何一类**。
真正该做的是把偏向讲明白、可配置、可评测，而不是埋一个阈值。

## 参数是在真实数据上调出来的

在 451 条官方界定结果上（检索层，绕过槽位抽取）：

| α | Top-1 | Ⅰ召回 | Ⅱ召回 | Ⅲ召回 | 成本/百例 |
|---|---|---|---|---|---|
| 0.00 | 62.3% | 73.8% | 71.1% | 36.8% | 88.7 |
| 0.05 | 62.1% | 67.2% | 72.1% | 40.8% | 84.0 |
| 0.10 | 61.2% | 61.5% | 68.6% | 48.8% | 79.2 |
| 0.20 | 57.9% | 53.3% | 60.8% | 57.6% | 74.5 |

成本口径：判低 = 3，判高 = 1，判对 = 0。

**α=0.05 几乎是免费的**：总体准确率只降 0.2pp，Ⅲ 类召回 +4.0pp，成本降 5.3%。
所以默认用它。需要更激进的漏检控制时再往上调，但要接受准确率的代价。

## 一个被证伪的假设（留在这里防止重复踩）

我原本以为问题在"扁平检索忽略了目录层级"，于是做了"先定子目录再定类别"的策略。
实测 **61.9% / Ⅲ 召回 32.8%**，比扁平检索还差 —— 假设不成立。

真正的原因更朴素：Ⅲ 类与 Ⅱ 类的区分常常依赖**临床用途的风险属性**，
而同为 IVD 试剂的两条文本在词面上几乎一样。纯词法检索在这上面有天花板，
换决策策略无法造出信号。要突破得靠向量检索或额外的领域特征。
"""

from __future__ import annotations

from dataclasses import dataclass

from mc_domain.enums import ManagementClass
from mc_domain.schemas import Candidate

_RANK = {ManagementClass.CLASS_I: 1, ManagementClass.CLASS_II: 2, ManagementClass.CLASS_III: 3}


@dataclass(frozen=True)
class RiskPreference:
    """风险偏好。`alpha` 是"就高"的带宽：分数落在最高分的 (1-α) 以上，就按更高的类别判。"""

    alpha: float
    label: str
    description: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1]，实际 {self.alpha}")


#: 预设。名字要能被非技术人员看懂——它会被写进审计日志和报告。
PRESETS: dict[str, RiskPreference] = {
    "strict": RiskPreference(
        alpha=0.0,
        label="严格按最高分",
        description="不做任何类别偏向。总体准确率最高，但 Ⅲ 类召回只有 36.8%。",
    ),
    "balanced": RiskPreference(
        alpha=0.05,
        label="均衡（默认）",
        description="几乎不损失准确率的前提下把 Ⅲ 类召回提高到 40.8%。",
    ),
    "risk_averse": RiskPreference(
        alpha=0.10,
        label="偏保守",
        description="Ⅲ 类召回 48.8%，代价是总体准确率降 1.1pp、Ⅰ 类召回明显下降。",
    ),
    "max_recall_iii": RiskPreference(
        alpha=0.20,
        label="最大 Ⅲ 类召回",
        description="Ⅲ 类召回 57.6%，但总体准确率降到 57.9%。适合只做初筛再全量人工复核的场景。",
    ),
}

DEFAULT_PRESET = "balanced"


def get_preference(name: str | None) -> RiskPreference:
    if not name:
        return PRESETS[DEFAULT_PRESET]
    key = name.strip().lower()
    if key not in PRESETS:
        raise ValueError(f"未知的风险偏好：{name}；可选：{', '.join(sorted(PRESETS))}")
    return PRESETS[key]


def _rank(candidate: Candidate) -> int:
    if candidate.management_class is None:
        return 0
    return _RANK.get(candidate.management_class, 0)


def select_candidate(
    candidates: list[Candidate], preference: RiskPreference
) -> tuple[Candidate | None, str]:
    """在候选里选出最终结论。

    返回 (选中的候选, 选择理由)。理由会进依据链——"为什么判 Ⅲ 类"必须能回答。
    """
    if not candidates:
        return None, "无候选"

    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = ordered[0]

    with_class = [c for c in ordered if c.management_class is not None]
    if not with_class:
        return top, "无可判类别，取最高分"

    anchor = with_class[0]
    if preference.alpha <= 0:
        return anchor, f"取最高分（{anchor.management_class.value} 类）"

    cutoff = anchor.score * (1 - preference.alpha)
    eligible = [c for c in with_class if c.score >= cutoff]
    chosen = max(eligible, key=lambda c: (_rank(c), c.score))

    if chosen is anchor:
        return anchor, (
            f"取最高分（{anchor.management_class.value} 类）；"
            f"带宽内无更高类别（α={preference.alpha:.2f}）"
        )

    return chosen, (
        f"就高不就低：候选分数落在最高分 {preference.alpha:.0%} 带宽内时取较高类别"
        f"（{anchor.management_class.value} → {chosen.management_class.value}），"
        f"依据《医疗器械分类规则》多类别择高原则"
    )


def available_presets() -> list[dict[str, object]]:
    """给 API / 前端列出可选策略。"""
    return [
        {"name": name, "alpha": p.alpha, "label": p.label, "description": p.description}
        for name, p in PRESETS.items()
    ]


__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "RiskPreference",
    "available_presets",
    "get_preference",
    "select_candidate",
]
