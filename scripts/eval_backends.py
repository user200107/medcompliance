"""三路检索对照：词法 / 向量 / 融合。

    python scripts/eval_backends.py --n 600
    python scripts/eval_backends.py --n 600 --weights 0.3,0.5,0.7

## 要回答的问题

词法检索（字符 bigram）在 Ⅲ 类召回上长期停在 36.8%~62.5%。
向量检索是唯一能触及"表述不同、含义相近"的手段——但它也可能毫无帮助
（如果 Ⅲ 与 Ⅱ 的差异本来就不在语义里）。所以必须用同一批样本量出来。

## 口径

- 与 `eval_retrieval.py` 相同的分层样本、相同的目录、相同的取舍策略。
- **查询向量一次性批量预热**：451 条查询按 10 条一批算完（46 次请求），
  检索时全部命中缓存，避免逐条 HTTP 把评测拖成十分钟。
- 索引从 `data/raw/catalog_vector_index` 加载，不重复计费。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in sorted((ROOT / "packages").glob("*/src")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from mc_domain.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mc_domain.catalog_loader import build_tree, load_csv  # noqa: E402
from mc_domain.decision import PRESETS, get_preference, select_candidate  # noqa: E402
from mc_domain.embeddings import (  # noqa: E402
    CachingEmbedder,
    DashScopeEmbeddings,
    HybridRetriever,
    VectorIndex,
    VectorRetriever,
)
from mc_domain.enums import ClassificationOutcome  # noqa: E402
from mc_domain.evaluation import class_metrics, load_cases, stratified_sample  # noqa: E402
from mc_domain.retrieval import KeywordCodeRetriever  # noqa: E402
from mc_domain.schemas import Candidate  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "raw" / "adjudications.jsonl"
DEFAULT_CATALOG = ROOT / "data" / "raw" / "catalog_real.csv"
DEFAULT_INDEX = ROOT / "data" / "raw" / "catalog_vector_index"


def collect(retriever, tree, text: str, topk: int) -> list[Candidate]:
    out: list[Candidate] = []
    for code, score in retriever.search_codes(text, limit=topk):
        node = tree.maybe_get(code)
        if node is None or node.management_class is None:
            continue
        out.append(
            Candidate(
                code=node.code,
                name=node.name,
                name_path=node.name_path,
                management_class=node.management_class,
                score=score,
            )
        )
    return out


def evaluate(hits_by_case: list[tuple], risk_name: str):
    pref = get_preference(risk_name)
    samples = []
    for case, hits in hits_by_case:
        chosen, _ = select_candidate(hits, pref)
        preds: list[ClassificationOutcome] = []
        if chosen is not None and chosen.management_class is not None:
            oc = ClassificationOutcome.from_management_class(chosen.management_class)
            if oc is not None:
                preds.append(oc)
        for cand in hits:
            oc = ClassificationOutcome.from_management_class(cand.management_class)
            if oc is not None and oc not in preds:
                preds.append(oc)
        samples.append((case.outcome, preds))
    return class_metrics(samples)


def print_row(label: str, m) -> None:
    print(
        f"{label:<22}"
        f"{m.top1_accuracy:>7.1%}"
        f"{m.top3_accuracy:>8.1%}"
        f"{m.class_prf(ClassificationOutcome.CLASS_I)[1]:>8.1%}"
        f"{m.class_prf(ClassificationOutcome.CLASS_II)[1]:>8.1%}"
        f"{m.class_prf(ClassificationOutcome.CLASS_III)[1]:>8.1%}"
        f"{m.macro_f1():>9.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="词法 / 向量 / 融合 三路对照")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--weights", default="0.3,0.5,0.7", help="融合时词法的权重")
    args = parser.parse_args()

    cases = [c for c in stratified_sample(load_cases(args.data), args.n) if c.outcome.in_class_scope]
    report = load_csv(args.catalog, catalog_version="backend-eval")
    report.raise_if_invalid()
    tree = build_tree(report.nodes, "backend-eval")
    print(f"类别范围内的分层样本：{len(cases)} 条；目录节点 {len(tree)}")

    gold = Counter(c.outcome.value for c in cases)
    majority, n_major = gold.most_common(1)[0]
    print(f"多数类基线（全判 {majority} 类）：{n_major / len(cases):.1%}；随机猜 33.3%\n")

    corpus = tree.search_corpus(aggregate_containers=True)
    lexical = KeywordCodeRetriever(corpus)

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"向量索引不存在：{index_path}（先跑 scripts/build_index.py）", file=sys.stderr)
        return 2
    index = VectorIndex.load(index_path)
    print(f"向量索引：{len(index)} 条 × {index.vectors.shape[1]} 维，模型 {index.meta.get('model')}")

    client = DashScopeEmbeddings(
        base_url=os.environ.get("MC_EMBEDDING_BASE_URL", ""),
        api_key=os.environ.get("MC_LLM_API_KEY", ""),
        model=os.environ.get("MC_EMBEDDING_MODEL", ""),
        dimensions=int(os.environ.get("MC_EMBEDDING_DIM", "1024") or 1024),
        max_chars=800,
    )
    cached = CachingEmbedder(client)

    print(f"批量预热查询向量（{len(cases)} 条，每批 10）…")
    added = cached.prewarm([c.input_text for c in cases])
    print(f"  预热 {added} 条，{cached.network_calls} 次请求，token {client.total_tokens}\n")

    vector = VectorRetriever(index, cached)

    headers = f"{'检索方式':<22}{'Top-1':>8}{'Top-3':>8}{'Ⅰ召回':>8}{'Ⅱ召回':>8}{'Ⅲ召回':>8}{'MacroF1':>9}"
    line = "=" * len(headers)

    for risk in ("strict", "balanced"):
        print(line)
        print(f"取舍策略：{PRESETS[risk].label}（α={PRESETS[risk].alpha:.2f}）")
        print(headers)
        print("-" * len(headers))

        hits = [(case, collect(lexical, tree, case.input_text, args.topk)) for case in cases]
        print_row("词法（字符 bigram）", evaluate(hits, risk))

        hits = [(case, collect(vector, tree, case.input_text, args.topk)) for case in cases]
        print_row("向量（text-embedding-v4）", evaluate(hits, risk))

        for raw_w in args.weights.split(","):
            w = float(raw_w)
            hybrid = HybridRetriever(lexical, vector, weight=w, pool=30)
            hits = [(case, collect(hybrid, tree, case.input_text, args.topk)) for case in cases]
            print_row(f"融合 词法={w:.1f}", evaluate(hits, risk))
        print()

    if vector.failures:
        print(f"⚠ 向量检索失败 {vector.failures} 次：{vector.last_error}", file=sys.stderr)

    print(
        "注：向量与融合的分数尺度不同（融合归一到 [0.5,1]），"
        "所以 α 带宽在两者上的松紧不一样——比较时看同一行内的相对变化。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
