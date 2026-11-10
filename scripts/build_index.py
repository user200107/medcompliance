"""为分类目录构建向量索引。

    python scripts/build_index.py
    python scripts/build_index.py --catalog data/raw/catalog_real.csv --max-chars 800

产物（`data/raw/catalog_vector_index/`）：
    vectors.npz   归一化后的向量矩阵
    codes.json    与行号一一对应的分类编码
    meta.json     模型、维度、节点数、构建时间——用于判断索引是否过期

**索引必须落盘**：重跑评测不该重复计费。构建一次约 190 次请求，
之后所有实验都从 npz 加载。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in sorted((ROOT / "packages").glob("*/src")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from mc_domain.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mc_domain.catalog_loader import build_tree, load_csv  # noqa: E402
from mc_domain.embeddings import (  # noqa: E402
    DashScopeEmbeddings,
    EmbeddingFailed,
    VectorIndex,
    build_corpus_texts,
)

DEFAULT_CATALOG = ROOT / "data" / "raw" / "catalog_real.csv"
DEFAULT_OUT = ROOT / "data" / "raw" / "catalog_vector_index"


def main() -> int:
    parser = argparse.ArgumentParser(description="构建分类目录的向量索引")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument(
        "--containers",
        action="store_true",
        default=True,
        help="容器节点带上子树词汇（与词法检索保持一致，默认开）",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MC_LLM_API_KEY", "")
    base_url = os.environ.get("MC_EMBEDDING_BASE_URL", "")
    model = os.environ.get("MC_EMBEDDING_MODEL", "")
    dim = int(os.environ.get("MC_EMBEDDING_DIM", "1024") or 1024)
    if not (api_key and base_url and model):
        print("缺少 MC_LLM_API_KEY / MC_EMBEDDING_BASE_URL / MC_EMBEDDING_MODEL", file=sys.stderr)
        return 2

    report = load_csv(args.catalog, catalog_version="index")
    report.raise_if_invalid()
    tree = build_tree(report.nodes, "index")

    pairs = tree.search_corpus(aggregate_containers=args.containers)
    codes, texts = build_corpus_texts(pairs, max_chars=args.max_chars)
    print(f"目录节点 {len(tree)}，参与向量化 {len(codes)} 条（截断 {args.max_chars} 字符）")

    fingerprint = hashlib.sha256(
        "\n".join(f"{c}\t{t}" for c, t in zip(codes, texts)).encode("utf-8")
    ).hexdigest()[:16]

    embedder = DashScopeEmbeddings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dimensions=dim,
        max_chars=args.max_chars,
    )

    print(f"开始向量化：{model}，{dim} 维，每批 10 条 …")
    try:
        vectors = embedder.embed(texts)
    except EmbeddingFailed as exc:
        print(f"向量化失败：{exc}", file=sys.stderr)
        return 1

    index = VectorIndex.build(
        codes,
        vectors,
        {
            "model": model,
            "dimensions": dim,
            "nodes": len(codes),
            "max_chars": args.max_chars,
            "containers_aggregated": args.containers,
            "corpus_fingerprint": fingerprint,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "embed_calls": embedder.calls,
            "embed_tokens": embedder.total_tokens,
        },
    )
    out_dir = index.save(args.out)
    print(
        f"完成：{len(index)} 条 × {index.vectors.shape[1]} 维 → {out_dir}\n"
        f"  调用 {embedder.calls} 次，token {embedder.total_tokens}，"
        f"指纹 {fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
