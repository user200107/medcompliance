"""隔离实验：只测「检索 + 目录 + 排序」能不能答对管理类别。

    python scripts/eval_retrieval.py --n 600
    python scripts/eval_retrieval.py --n 300 --rerank --doc-mode compact
    python scripts/eval_retrieval.py --n 600 --risks strict,balanced,risk_averse,max_recall_iii

## 为什么需要这个脚本

端到端评测的覆盖率只有 1.7%——绝大多数样本在**槽位抽取**阶段就转成了追问，
根本走不到检索。于是「目录质量」「检索质量」完全测不出来。

这个脚本把抽取层摘掉，直接拿官方原生的「产品名称 + 产品详情」去检索目录。

## 重排序对照（--rerank）

在**排序质量**上对比四条路径，全部用 `strict`（α=0）以避免风险旋钮干扰：

    单段      字符 bigram 的原始顺序
    纯重排    完全按 rerank 分数重排
    融合 w    按 w·base + (1-w)·rerank 混合（w 是 base 的权重）

**为什么用 α=0 做对比**：α 是"往高类别偏"的旋钮，会改变类别分布。
如果拿它和重排序混在一起，就分不清"排序变好了"和"结论被推向 Ⅲ 类了"。
排序质量必须单独测。

## 实测结论（2026-11，n=90，qwen3-rerank）

- 纯重排 Top-1 **63.3%**，单段 **70.0%**（−6.7pp）；compact 模式同样 −4.4pp。
- 两者的 Top-3 都是 96.7% —— 重排不改变召回，符合预期。
- 但 Ⅰ 类召回从 75.0% 塌到 41.7%，Ⅲ 类从 56.0% 升到 68.0%。

即：**重排把结论系统性地推向高类别，而没有提升排序准确率。**
而"往高类别偏"这件事，`risk_averse` / `max_recall_iii` 已经能确定性做到，且不花钱。
所以 P1 默认不启用重排序，客户端保留在 `mc_domain/rerank.py` 里以备 P2 使用。

这个负结果是刻意记录的：不写下来，下一个人（或三个月后的我）会再花一次 token 重跑。
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
from mc_domain.enums import ClassificationOutcome  # noqa: E402
from mc_domain.evaluation import class_metrics, load_cases, stratified_sample  # noqa: E402
from mc_domain.rerank import (  # noqa: E402
    DashScopeReranker,
    RerankFailed,
    build_doc_text,
)
from mc_domain.retrieval import KeywordCodeRetriever  # noqa: E402
from mc_domain.schemas import Candidate  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "raw" / "adjudications.jsonl"
DEFAULT_CATALOG = ROOT / "data" / "raw" / "catalog_real.csv"


def recall_candidates(retriever, tree, text: str, k: int) -> list[Candidate]:
    out: list[Candidate] = []
    for code, score in retriever.search_codes(text, limit=k):
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



def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def reorder(recall: list[Candidate], scores: dict[str, float], limit: int) -> list[Candidate]:
    ordered = sorted(recall, key=lambda c: scores.get(c.code, -1.0), reverse=True)
    out = [
        Candidate(
            code=c.code,
            name=c.name,
            name_path=c.name_path,
            management_class=c.management_class,
            score=scores.get(c.code, 0.0),
        )
        for c in ordered
    ]
    return out[:limit]


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
        f"{label:<24}"
        f"{m.top1_accuracy:>7.1%}"
        f"{m.top3_accuracy:>8.1%}"
        f"{m.class_prf(ClassificationOutcome.CLASS_I)[1]:>8.1%}"
        f"{m.class_prf(ClassificationOutcome.CLASS_II)[1]:>8.1%}"
        f"{m.class_prf(ClassificationOutcome.CLASS_III)[1]:>8.1%}"
        f"{m.macro_f1():>9.3f}"
    )


def print_header(title: str) -> None:
    header = f"{'路径':<24}{'Top-1':>8}{'Top-3':>8}{'Ⅰ召回':>8}{'Ⅱ召回':>8}{'Ⅲ召回':>8}{'MacroF1':>9}"
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(description="检索层隔离评测")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--risks", default="balanced")
    parser.add_argument("--rerank", action="store_true", help="开启重排序并给出排序质量对照")
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--doc-mode", default="compact", choices=["compact", "full"])
    parser.add_argument("--fusion-weights", default="0.3,0.5,0.7", help="base 权重，逗号分隔")
    args = parser.parse_args()

    risk_names = [n.strip() for n in args.risks.split(",") if n.strip()]

    cases = [c for c in stratified_sample(load_cases(args.data), args.n) if c.outcome.in_class_scope]
    report = load_csv(args.catalog, catalog_version="retrieval-eval")
    report.raise_if_invalid()
    tree = build_tree(report.nodes, "retrieval-eval")

    corpus = [(n.code, n.search_blob) for n in tree if n.management_class and n.search_blob]
    base = KeywordCodeRetriever(corpus)
    print(f"类别范围内的分层样本：{len(cases)} 条；检索语料 {len(corpus)} 个可定类节点")

    base_hits = [(case, recall_candidates(base, tree, case.input_text, args.topk)) for case in cases]
    gold_counts = Counter(c.outcome.value for c in cases)
    majority, majority_n = gold_counts.most_common(1)[0]
    print(f"多数类基线（全判为 {majority} 类）：{majority_n / len(cases):.1%}；随机猜 33.3%")

    print_header("单段检索（字符 bigram），按风险偏好")
    for name in risk_names:
        print_row(PRESETS[name].label, evaluate(base_hits, name))

    if args.rerank:
        api_key = os.environ.get("MC_LLM_API_KEY", "")
        model = os.environ.get("MC_RERANK_MODEL", "")
        base_url = os.environ.get("MC_RERANK_BASE_URL", "")
        if not (api_key and model and base_url):
            print(
                "\n未配置重排序（需要 MC_LLM_API_KEY / MC_RERANK_MODEL / MC_RERANK_BASE_URL），已跳过。",
                file=sys.stderr,
            )
            return 1

        code_text = {
            n.code: build_doc_text(n, args.doc_mode) for n in tree if n.management_class
        }
        reranker = DashScopeReranker(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=int(os.environ.get("MC_LLM_TIMEOUT_S", "30") or 30),
            max_retries=2,
        )
        weights = [float(w) for w in args.fusion_weights.split(",") if w.strip()]

        print(
            f"\n跑重排序：{model}，召回 {args.recall_k} 条/query，"
            f"文档模式 {args.doc_mode}，融合权重 {weights} …"
        )

        pure_hits: list[tuple] = []
        fused_hits: dict[float, list[tuple]] = {w: [] for w in weights}
        total_tokens = 0
        failures = 0

        for i, case in enumerate(cases, 1):
            recall = recall_candidates(base, tree, case.input_text, args.recall_k)
            if not recall:
                pure_hits.append((case, []))
                for w in weights:
                    fused_hits[w].append((case, []))
                continue

            docs = [code_text.get(c.code, c.code) for c in recall]
            try:
                order = reranker.rerank(case.input_text, docs, top_n=len(docs))
            except RerankFailed as exc:
                failures += 1
                print(f"  ! {case.product_name[:20]} 重排序失败，已降级：{exc}", file=sys.stderr)
                pure_hits.append((case, recall[: args.topk]))
                for w in weights:
                    fused_hits[w].append((case, recall[: args.topk]))
                continue

            if reranker.last_usage:
                total_tokens += int(reranker.last_usage.get("total_tokens") or 0)

            rr_by_code = {recall[idx].code: score for idx, score in order if 0 <= idx < len(recall)}
            pure_hits.append((case, reorder(recall, rr_by_code, args.topk)))

            base_norm = _minmax([c.score for c in recall])
            rr_norm = _minmax([rr_by_code.get(c.code, 0.0) for c in recall])
            for w in weights:
                fused = {
                    c.code: w * base_norm[j] + (1 - w) * rr_norm[j] for j, c in enumerate(recall)
                }
                fused_hits[w].append((case, reorder(recall, fused, args.topk)))

            if i % 50 == 0:
                print(f"  {i}/{len(cases)}（累计 token {total_tokens}）")

        print_header("排序质量对照（全部 α=0，只看排序本身）")
        print_row("单段（baseline）", evaluate(base_hits, "strict"))
        print_row("纯重排", evaluate(pure_hits, "strict"))
        for w in weights:
            print_row(f"融合 base={w:.1f}", evaluate(fused_hits[w], "strict"))

        print("\n同样的对照，但套用风险偏好 max_recall_iii（α=0.20）：")
        print_header("")
        print_row("单段 + α=0.20", evaluate(base_hits, "max_recall_iii"))
        print_row("纯重排 + α=0.20", evaluate(pure_hits, "max_recall_iii"))

        print(f"\n重排序累计 token：{total_tokens}；失败 {failures} 次")

    print(
        "\n注：重排序不改变召回，所以 Top-3 应基本不动；它的作用点是 Top-1。"
        "\n    α 与重排序都会改变类别分布，混在一起就分不清是排序变好还是结论被推向高类别。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
