"""诊断并优化 Ⅲ 类召回。

    python scripts/tune_class_decision.py --n 600 --catalog data/raw/catalog_real.csv

## 出发点

检索层现状：Top-1 62.3%，**Top-3 98.9%**。
这个落差说明正确类别几乎总在前三里被检索到——问题不在"找不到"，而在**最终选谁**。

而各类别表现极不均匀：Ⅲ 类召回 36.8%，Ⅰ 类 73.8%，Ⅱ 类 71.1%。
Ⅲ 类恰恰是最贵的一类错误：Ⅱ 类走省级注册、Ⅲ 类走国家局审评，
把 Ⅲ 类判成 Ⅱ 类是"完全走错流程"，不是"稍微不准"。

所以这个脚本做两件事：
1. **诊断**：错在哪——是排序被多类挤掉，还是某些子目录系统性失手？
2. **对照实验**：几种决策策略 + 成本矩阵，选一个能说清 trade-off 的。

注意：这不是搜参。策略必须能讲出领域理由，否则调出来的数字换一批数据就崩。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in sorted((ROOT / "packages").glob("*/src")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from mc_domain.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mc_domain.catalog_loader import build_tree, load_csv  # noqa: E402
from mc_domain.enums import ClassificationOutcome, ManagementClass  # noqa: E402
from mc_domain.evaluation import (  # noqa: E402
    ExpectedOutcome,
    load_cases,
    stratified_sample,
    wilson_interval,
)
from mc_domain.retrieval import KeywordCodeRetriever  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "raw" / "adjudications.jsonl"
DEFAULT_CATALOG = ROOT / "data" / "raw" / "catalog_real.csv"

#: 成本矩阵：漏判（判低）远比误判（判高）贵。
#: 理由不是拍脑袋——Ⅱ 类省级注册 vs Ⅲ 类国家局审评是两条完全不同的流程，
#: 而且 Ⅲ 类是高风险植入/生命支持类，低估的后果不可承受。
COST_UNDER = 3   # 判低：漏掉高风险
COST_OVER = 1    # 判高：多跑一趟审评
COST_RIGHT = 0

_RANK = {"I": 1, "II": 2, "III": 3}


@dataclass(frozen=True)
class Hit:
    code: str
    score: float
    cls: ClassificationOutcome
    sub: str          # 子目录编码，用于诊断


def rank_of(cls: str) -> int:
    return _RANK[cls]


# ---------------------------------------------------------------------------
# 决策策略
# ---------------------------------------------------------------------------


def policy_top1(hits: list[Hit]) -> ClassificationOutcome | None:
    """现状：直接取检索分最高的那个节点所属类别。"""
    return hits[0].cls if hits else None


def policy_class_max(hits: list[Hit]) -> ClassificationOutcome | None:
    """按类别聚合：取"该类最高分"作为该类的代表分，再跨类取最大。

    与 top1 的差别：避免单个节点的高分把整个类别带偏。
    """
    best: dict[ClassificationOutcome, float] = {}
    for h in hits:
        if h.score > best.get(h.cls, -1.0):
            best[h.cls] = h.score
    if not best:
        return None
    return max(best.items(), key=lambda kv: kv[1])[0]


def policy_class_sum2(hits: list[Hit]) -> ClassificationOutcome | None:
    """按类别聚合：取该类前 2 名分数之和。类别内有多条命中时更稳。"""
    acc: dict[ClassificationOutcome, list[float]] = defaultdict(list)
    for h in hits:
        acc[h.cls].append(h.score)
    if not acc:
        return None
    return max(acc.items(), key=lambda kv: sum(sorted(kv[1], reverse=True)[:2]))[0]


def make_band_policy(alpha: float, base: str = "top1"):
    """风险偏向策略：在最优分的一个带宽内，选**更高**的管理类别。

    这不是随手加的启发式，它对应《医疗器械分类规则》的"就高不就低"原则：
    一个产品若可能同时落入多个类别，按较高类别管理。

    alpha=0 等价于不偏向；alpha 越大越激进地把结论往高类别推。
    """

    def policy(hits: list[Hit]) -> ClassificationOutcome | None:
        if not hits:
            return None
        if base == "top1":
            anchor_score = hits[0].score
        else:
            acc: dict[ClassificationOutcome, list[float]] = defaultdict(list)
            for h in hits:
                acc[h.cls].append(h.score)
            per_class = {c: sum(sorted(v, reverse=True)[:2]) for c, v in acc.items()}
            anchor_score = max(per_class.values())
            anchor_cls = max(per_class.items(), key=lambda kv: kv[1])[0]
            if alpha <= 0:
                return anchor_cls

        cutoff = anchor_score * (1 - alpha)
        eligible = [h for h in hits if h.score >= cutoff]
        if not eligible:
            return hits[0].cls
        # 取带宽内类别最高者；同类别取分最高
        return max(eligible, key=lambda h: (rank_of(h.cls.value), h.score)).cls

    return policy


def make_prior_policy(beta: float):
    """按类别频次做先验校正：score / freq^beta。

    目录里 Ⅱ 类节点占 48%，词法检索会天然多命中 Ⅱ 类。
    beta>0 用于抵消这种"语料越多越容易被选中"的偏差。
    """

    def policy_factory(freq: Counter[str]):
        def policy(hits: list[Hit]) -> ClassificationOutcome | None:
            best: dict[ClassificationOutcome, float] = {}
            for h in hits:
                adj = h.score / (freq.get(h.cls.value, 1) ** beta)
                if adj > best.get(h.cls, -1.0):
                    best[h.cls] = adj
            if not best:
                return None
            return max(best.items(), key=lambda kv: kv[1])[0]

        return policy

    return policy_factory


def make_subtree_policy(alpha: float = 0.0, use_sum: bool = True):
    """尊重目录层级：先定子目录，再在子目录内定类别。

    扁平检索的隐含假设是"所有节点可比"，但分类目录本来就是一棵树：
    先判断"这是哪一类器械"再看具体编码，本来就是注册专员的实际思路。
    实测里 22 子目录（临床检验/IVD）占误判的 45%，正是扁平检索
    把不同子目录的相似表述混在一起的典型症状。
    """

    def policy(hits: list[Hit]) -> ClassificationOutcome | None:
        if not hits:
            return None

        sub_scores: dict[str, float] = defaultdict(float)
        for h in hits:
            sub_scores[h.sub] = sub_scores[h.sub] + h.score if use_sum else max(
                sub_scores[h.sub], h.score
            )
        best_sub = max(sub_scores.items(), key=lambda kv: kv[1])[0]

        within = [h for h in hits if h.sub == best_sub]
        if not within:
            return hits[0].cls

        per_class: dict[ClassificationOutcome, float] = {}
        for h in within:
            per_class[h.cls] = max(per_class.get(h.cls, -1.0), h.score)
        if not per_class:
            return within[0].cls

        anchor_score = max(per_class.values())
        if alpha <= 0:
            return max(per_class.items(), key=lambda kv: kv[1])[0]

        cutoff = anchor_score * (1 - alpha)
        eligible = [c for c, s in per_class.items() if s >= cutoff]
        return max(eligible, key=lambda c: rank_of(c.value))

    return policy


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    name: str
    correct: int = 0
    n: int = 0
    per_gold_hit: dict[str, int] = None  # type: ignore[assignment]
    per_gold_total: dict[str, int] = None  # type: ignore[assignment]
    cost: int = 0

    def __post_init__(self) -> None:
        self.per_gold_hit = defaultdict(int)
        self.per_gold_total = defaultdict(int)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def recall(self, cls: str) -> float:
        total = self.per_gold_total.get(cls, 0)
        return self.per_gold_hit.get(cls, 0) / total if total else 0.0

    @property
    def mean_cost(self) -> float:
        return self.cost / self.n if self.n else 0.0


def evaluate(name: str, samples, policy) -> PolicyResult:
    res = PolicyResult(name=name)
    for gold, hits in samples:
        pred = policy(hits)
        res.n += 1
        gold_cls = gold.value
        res.per_gold_total[gold_cls] += 1
        if pred is None:
            res.cost += COST_UNDER
            continue
        pred_cls = pred.value
        if pred_cls == gold_cls:
            res.correct += 1
            res.per_gold_hit[gold_cls] += 1
            res.cost += COST_RIGHT
        elif rank_of(pred_cls) < rank_of(gold_cls):
            res.cost += COST_UNDER
        else:
            res.cost += COST_OVER
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Ⅲ 类召回诊断与决策策略对照")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    cases = [c for c in stratified_sample(load_cases(args.data), args.n) if c.outcome.in_class_scope]
    report = load_csv(args.catalog, catalog_version="tune")
    report.raise_if_invalid()
    tree = build_tree(report.nodes, "tune")

    corpus = [(n.code, n.search_blob) for n in tree if n.management_class and n.search_blob]
    retriever = KeywordCodeRetriever(corpus)

    from mc_domain.enums import ClassificationOutcome as CO

    def cls_of(node) -> CO | None:
        return CO.from_management_class(node.management_class)

    freq: Counter[str] = Counter(
        n.management_class.value for n in tree if n.management_class is not None
    )

    samples: list[tuple[ExpectedOutcome, list[Hit]]] = []
    rank_of_gold: Counter[int] = Counter()
    sub_errors: Counter[str] = Counter()
    sub_totals: Counter[str] = Counter()

    for case in cases:
        raw = retriever.search_codes(case.input_text, limit=args.topk)
        hits: list[Hit] = []
        for code, score in raw:
            node = tree.maybe_get(code)
            if node is None or node.management_class is None:
                continue
            outcome = cls_of(node)
            if outcome is None:
                continue
            sub = code.split("-")[0]
            hits.append(Hit(code=code, score=score, cls=outcome, sub=sub))

        samples.append((case.outcome, hits))
        gold_cls = case.outcome.value
        rank = next((i + 1 for i, h in enumerate(hits) if h.cls.value == gold_cls), 0)
        rank_of_gold[rank] += 1
        top_sub = hits[0].sub if hits else "-"
        sub_totals[top_sub] += 1
        if not hits or hits[0].cls.value != gold_cls:
            sub_errors[top_sub] += 1

    print(f"样本 {len(samples)} 条，检索语料 {len(corpus)} 节点")
    print(f"目录类别频次：{dict(freq)}")
    print("\n=== 诊断：正确类别在检索结果里的名次分布 ===")
    for rank in sorted(rank_of_gold)[:8]:
        label = f"第 {rank} 名" if rank else "未召回"
        print(f"  {label:<8} {rank_of_gold[rank]:5d}  {rank_of_gold[rank] / len(samples):6.1%}")
    in_top3 = sum(v for k, v in rank_of_gold.items() if 1 <= k <= 3)
    print(f"  → 前 3 名内命中率 {in_top3 / len(samples):.1%}：类别基本都找得到，问题在取舍")

    print("\n=== 诊断：哪些子目录最容易被误选为 Top-1 ===")
    for sub, total in sub_errors.most_common(8):
        print(f"  子目录 {sub:<3} 误判 {sub_errors[sub]:3d} / 出现 {sub_totals[sub]:3d}")

    # ---- 策略对照 ----
    print("\n=== 策略对照（成本：判低=3，判高=1）===")
    results: list[PolicyResult] = [
        evaluate("① top1（现状）", samples, policy_top1),
        evaluate("② 类别取最高分", samples, policy_class_max),
        evaluate("③ 类别取前2名之和", samples, policy_class_sum2),
    ]
    for alpha in (0.02, 0.05, 0.10, 0.20):
        results.append(
            evaluate(f"④ 就高不就低 α={alpha:.2f}", samples, make_band_policy(alpha))
        )
    for beta in (0.2, 0.5, 1.0):
        results.append(
            evaluate(f"⑤ 频次校正 β={beta:.1f}", samples, make_prior_policy(beta)(freq))
        )
    results.append(evaluate("⑥ 先定子目录", samples, make_subtree_policy(0.0)))
    for alpha in (0.05, 0.10, 0.20):
        results.append(
            evaluate(f"⑦ 子目录+就高 α={alpha:.2f}", samples, make_subtree_policy(alpha))
        )

    header = f"{'策略':<26}{'Top-1':>8}{'Ⅰ召回':>8}{'Ⅱ召回':>8}{'Ⅲ召回':>8}{'成本/百例':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        lo, hi = wilson_interval(r.correct, r.n)
        print(
            f"{r.name:<26}"
            f"{r.accuracy:>7.1%}"
            f"{r.recall('I'):>8.1%}"
            f"{r.recall('II'):>8.1%}"
            f"{r.recall('III'):>8.1%}"
            f"{r.mean_cost * 100:>11.1f}"
        )
        _ = (lo, hi)

    best_acc = max(results, key=lambda r: r.accuracy)
    best_cost = min(results, key=lambda r: r.mean_cost)
    print(f"\n准确率最高：{best_acc.name}（{best_acc.accuracy:.1%}）")
    print(f"成本最低：  {best_cost.name}（{best_cost.mean_cost * 100:.1f}/百例）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
