"""向量检索：把目录节点与查询都映射到同一空间，用余弦相似度排序。

## 为什么需要它

词法检索（字符 bigram）在 Ⅲ 类召回上撞了天花板：Ⅲ 与 Ⅱ 的区分常常依赖
临床用途的风险属性，而**同一子目录下两条描述的词面几乎一样**。
词法拿不到这层信号，于是 Ⅲ 类召回长期停在 36.8%~62.5%。

向量检索不保证能解决——如果词面都一样，语义也可能一样——但它是唯一
能触及"表述不同、含义相近"的手段，所以必须试，并且要用同一台评测机量出来。

## 设计取舍

- **索引放内存**：1,689 个可定类节点 × 1,024 维 float32 ≈ 7 MB。
  一个人维护的项目上 Milvus 是负担，numpy 点积足够快（一次查询 <1 ms）。
- **向量先做 L2 归一化**：余弦相似度退化为点积，省一次除法。
- **批大小 10**：百炼的 embedding 接口每批最多 10 条，超了直接报错。
- **索引落盘**：重新评测不该重复计费。`build_index.py` 存 npz，评测直接加载。

## 融合而不是替换

纯向量检索在"产品名与品名举例字面一致"的场景上会输给 bigram——
而这恰恰是器械目录里很常见的一类。所以提供 `HybridRetriever`：
两路各自 min-max 归一化后加权求和。

**刻意不用 RRF**：RRF 的分数是 1/(k+rank)，相邻名次的比值接近 1，
会把 `decision.py` 里"就高不就低"的 α 带宽判据撑爆（几乎所有候选都落进带宽）。
分数级融合能保持相对差异，α 的语义才不变。
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from mc_domain.env import http_client_kwargs
from mc_domain.retrieval import KeywordCodeRetriever

logger = logging.getLogger(__name__)

#: 百炼 embedding 接口每批上限
MAX_BATCH = 10


class EmbeddingFailed(RuntimeError):
    """向量化失败。调用方应降级到词法检索并留痕，不得当作成功。"""


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class DashScopeEmbeddings:
    """OpenAI 兼容的 /embeddings 客户端（百炼 text-embedding-v4 实测可用）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        dimensions: int = 1024,
        timeout_s: int = 30,
        max_retries: int = 3,
        max_chars: int = 800,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("embedding 需要 base_url 与 api_key")
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._model = model
        self._dim = dimensions
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._max_chars = max_chars
        self.total_tokens = 0
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        out: list[list[float]] = []
        with httpx.Client(**http_client_kwargs(self._timeout)) as client:  # type: ignore[arg-type]
            for start in range(0, len(texts), MAX_BATCH):
                batch = [t[: self._max_chars] for t in texts[start : start + MAX_BATCH]]
                out.extend(self._embed_batch(client, batch))
        return out

    def _embed_batch(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": batch,
            "dimensions": self._dim,
            "encoding_format": "float",
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = client.post(self._url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
                self.calls += 1
                usage = body.get("usage") or {}
                self.total_tokens += int(usage.get("total_tokens") or 0)

                data = sorted(body["data"], key=lambda item: item.get("index", 0))
                vectors = [item["embedding"] for item in data]
                if len(vectors) != len(batch):
                    raise EmbeddingFailed(
                        f"返回条数不符：请求 {len(batch)} 条，返回 {len(vectors)} 条"
                    )
                for vec in vectors:
                    if len(vec) != self._dim:
                        raise EmbeddingFailed(
                            f"维度不符：期望 {self._dim}，实际 {len(vec)}"
                        )
                return vectors
            except EmbeddingFailed:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("向量化第 %d 次失败：%s", attempt + 1, exc)

        raise EmbeddingFailed(f"向量化失败（已重试 {self._max_retries} 次）：{last_error}")


class CachingEmbedder:
    """带缓存的包装器，核心用途是**把查询批量预热**。

    `VectorRetriever.search_codes` 一次只向量化一条查询，逐条调用会变成
    「451 条样本 = 451 次 HTTP」。套上这一层后可以先把所有查询按批（10 条）
    一次性算完，检索时全部命中缓存。

    缓存无上限：一次评测的查询集只有几百条，不需要淘汰策略。
    """

    def __init__(self, client: EmbeddingClient, *, batch_size: int = MAX_BATCH) -> None:
        self.client = client
        self.batch_size = batch_size
        self.cache: dict[str, list[float]] = {}
        self.network_calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self.client.embed(batch)
            self.network_calls += 1
            for text, vec in zip(batch, vectors):
                self.cache[text] = vec
        return [self.cache[t] for t in texts]

    def prewarm(self, texts: Sequence[str]) -> int:
        """预先批量算好，返回本次新增的缓存条目数。"""
        before = len(self.cache)
        self.embed(texts)
        return len(self.cache) - before


@dataclass
class VectorIndex:
    codes: list[str]
    vectors: np.ndarray  # (n, dim) float32, 已 L2 归一化
    meta: dict[str, object]

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors 应为二维矩阵，实际 {self.vectors.shape}")
        if len(self.codes) != self.vectors.shape[0]:
            raise ValueError(
                f"codes 与 vectors 行数不符：{len(self.codes)} vs {self.vectors.shape[0]}"
            )

    def __len__(self) -> int:
        return len(self.codes)

    @classmethod
    def build(cls, codes: Sequence[str], vectors: Sequence[Sequence[float]], meta: dict) -> VectorIndex:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return cls(codes=list(codes), vectors=matrix / norms, meta=dict(meta))

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self.vectors)
        (directory / "codes.json").write_text(
            json.dumps(self.codes, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> VectorIndex:
        directory = Path(directory)
        with np.load(directory / "vectors.npz") as data:
            vectors = data["vectors"]
        codes = json.loads((directory / "codes.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(codes=codes, vectors=vectors, meta=meta)


class VectorRetriever:
    """实现与 `KeywordCodeRetriever` 相同的接口，可原地替换。"""

    def __init__(self, index: VectorIndex, embedder: EmbeddingClient) -> None:
        self.index = index
        self.embedder = embedder
        self._row_of = {code: i for i, code in enumerate(index.codes)}
        self.failures = 0
        self.last_error: str | None = None

    def search_codes(
        self, query: str, *, limit: int, allowed_codes: set[str] | None = None
    ) -> list[tuple[str, float]]:
        try:
            query_vec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        except EmbeddingFailed as exc:
            # 查询向量化失败：没有退路可用（词法索引不在这里），留痕并返回空
            self.failures += 1
            self.last_error = str(exc)
            logger.warning("查询向量化失败：%s", exc)
            return []

        norm = float(np.linalg.norm(query_vec))
        if norm > 0:
            query_vec = query_vec / norm

        scores = self.index.vectors @ query_vec
        if allowed_codes is not None:
            mask = np.array([code in allowed_codes for code in self.index.codes])
            scores = np.where(mask, scores, -np.inf)

        k = min(limit, len(self.index.codes))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            (self.index.codes[i], float(scores[i]))
            for i in top_idx
            if math.isfinite(float(scores[i]))
        ]


class HybridRetriever:
    """词法 + 向量的分数级融合。

    `weight` 是词法的权重（0 = 纯向量，1 = 纯词法）。两路各自在**本次查询的
    候选集内**做 min-max 归一化后加权求和。缺一路的候选按 0 计。
    """

    def __init__(
        self,
        lexical: KeywordCodeRetriever,
        vector: VectorRetriever,
        *,
        weight: float = 0.5,
        pool: int = 30,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight 必须在 [0,1]，实际 {weight}")
        self.lexical = lexical
        self.vector = vector
        self.weight = weight
        self.pool = pool

    def search_codes(
        self, query: str, *, limit: int, allowed_codes: set[str] | None = None
    ) -> list[tuple[str, float]]:
        lex_hits = self.lexical.search_codes(query, limit=self.pool, allowed_codes=allowed_codes)
        vec_hits = self.vector.search_codes(query, limit=self.pool, allowed_codes=allowed_codes)
        if not lex_hits and not vec_hits:
            return []
        if not lex_hits:
            return vec_hits[:limit]
        if not vec_hits:
            return lex_hits[:limit]

        lex = dict(lex_hits)
        vec = dict(vec_hits)
        lex_norm = _minmax(lex)
        vec_norm = _minmax(vec)

        fused = {
            code: self.weight * lex_norm.get(code, 0.0) + (1 - self.weight) * vec_norm.get(code, 0.0)
            for code in set(lex) | set(vec)
        }
        # 融合分为 0 说明两路都没有给它任何贡献（只出现在权重为 0 的那一路），
        # 这种候选不是真候选，直接丢掉。否则 weight=1 会混进纯向量召回的条目。
        fused = {code: score for code, score in fused.items() if score > 0}

        # 平局时按"加权名次"决定，而不是靠 set 的迭代顺序。
        # 否则 weight=1 与 weight=0 的退化行为无法复现（单测直接抓到了这一点）。
        lex_rank = {code: i for i, code in enumerate(lex)}
        vec_rank = {code: i for i, code in enumerate(vec)}
        big = self.pool + 1
        tie_break = {
            code: self.weight * lex_rank.get(code, big) + (1 - self.weight) * vec_rank.get(code, big)
            for code in fused
        }
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], tie_break[kv[0]]))
        return ranked[:limit]


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """把一路的分数映射到 **[0.5, 1]**，而不是 [0, 1]。

    为什么不映射到 0：融合时"某一路没有这个候选"是按 0 计分的。
    如果归一化也把该路的**最后一名**压成 0，它就会和"完全没被召回"
    撞成同一个分数，排序变成随机的（单测直接暴露过这个问题）。
    抬到 [0.5, 1] 保证：被任一路召回过，就一定排在"两路都没召回"之前。
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: 0.5 + 0.5 * (v - lo) / (hi - lo) for k, v in scores.items()}


def build_corpus_texts(
    pairs: Iterable[tuple[str, str]], *, max_chars: int = 800
) -> tuple[list[str], list[str]]:
    """(codes, texts)，按 max_chars 截断以控制计费。"""
    codes: list[str] = []
    texts: list[str] = []
    for code, text in pairs:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        codes.append(code)
        texts.append(cleaned[:max_chars])
    return codes, texts


__all__ = [
    "MAX_BATCH",
    "DashScopeEmbeddings",
    "EmbeddingClient",
    "EmbeddingFailed",
    "HybridRetriever",
    "VectorIndex",
    "VectorRetriever",
    "build_corpus_texts",
]
