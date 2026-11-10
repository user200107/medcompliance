"""向量检索的单测（全部离线，用假 embedder）。

重点不是"能不能算余弦"，而是三条容易悄悄坏掉的约定：
1. 索引构建时做了 L2 归一化（否则点积不是余弦，排序会偏）；
2. `allowed_codes` 过滤必须生效（粗筛阶段只允许子目录）；
3. 融合权重 1.0 / 0.0 必须分别退化为纯词法 / 纯向量。
"""

from __future__ import annotations

import numpy as np
import pytest

from mc_domain.embeddings import (
    EmbeddingFailed,
    HybridRetriever,
    VectorIndex,
    VectorRetriever,
    _minmax,
    build_corpus_texts,
)
from mc_domain.retrieval import KeywordCodeRetriever


class FakeEmbedder:
    """按关键词构造三维向量，便于断言"谁更近"。"""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            if t in self.mapping:
                out.append(self.mapping[t])
            elif "眼科" in t:
                out.append([1.0, 0.0, 0.0])
            elif "手术" in t:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


class ExplodingEmbedder:
    def embed(self, texts):
        raise EmbeddingFailed("模拟失败")


@pytest.fixture
def index():
    return VectorIndex.build(
        codes=["eye", "surgery", "other"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        meta={"model": "fake"},
    )


def test_build_normalizes_vectors(index):
    norms = np.linalg.norm(index.vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_search_returns_nearest(index):
    retriever = VectorRetriever(index, FakeEmbedder())
    hits = retriever.search_codes("眼科超声诊断设备", limit=3)
    assert hits[0][0] == "eye"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_allowed_codes_filters(index):
    retriever = VectorRetriever(index, FakeEmbedder())
    hits = retriever.search_codes("眼科超声", limit=3, allowed_codes={"surgery", "other"})
    assert {c for c, _ in hits} == {"surgery", "other"}


def test_embedding_failure_is_recorded(index):
    retriever = VectorRetriever(index, ExplodingEmbedder())
    assert retriever.search_codes("任意", limit=3) == []
    assert retriever.failures == 1
    assert retriever.last_error and "模拟失败" in retriever.last_error


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.codes == index.codes
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.meta["model"] == "fake"


def test_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="行数不符"):
        VectorIndex(codes=["a"], vectors=np.zeros((2, 3), dtype=np.float32), meta={})


def test_minmax_edge_cases():
    assert _minmax({}) == {}
    assert _minmax({"a": 1.0, "b": 1.0}) == {"a": 1.0, "b": 1.0}
    # 映射到 [0.5, 1]：保证"被召回过"永远排在"没被召回"（按 0 计）之前
    out = _minmax({"a": 2.0, "b": 4.0})
    assert out == {"a": 0.5, "b": 1.0}


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------


def _lexical():
    return KeywordCodeRetriever(
        [
            ("eye", "眼科超声诊断设备"),
            ("surgery", "超声手术设备"),
            ("other", "物理治疗器械"),
        ]
    )


def test_hybrid_weight_one_equals_lexical_order(index):
    hybrid = HybridRetriever(_lexical(), VectorRetriever(index, FakeEmbedder()), weight=1.0)
    lex = _lexical().search_codes("眼科超声诊断设备", limit=3)
    hits = hybrid.search_codes("眼科超声诊断设备", limit=3)
    assert [c for c, _ in hits] == [c for c, _ in lex]


def test_hybrid_weight_zero_equals_vector_order(index):
    hybrid = HybridRetriever(_lexical(), VectorRetriever(index, FakeEmbedder()), weight=0.0)
    vec = VectorRetriever(index, FakeEmbedder()).search_codes("眼科超声诊断设备", limit=3)
    hits = hybrid.search_codes("眼科超声诊断设备", limit=3)
    assert [c for c, _ in hits] == [c for c, _ in vec]


def test_hybrid_keeps_fused_scores_in_unit_range(index):
    """融合分数必须落在 [0,1]，否则 decision 的 α 带宽判据会失真。"""
    hybrid = HybridRetriever(_lexical(), VectorRetriever(index, FakeEmbedder()), weight=0.5)
    hits = hybrid.search_codes("眼科超声诊断设备", limit=3)
    assert all(0.0 <= score <= 1.0 for _, score in hits)


def test_hybrid_rejects_bad_weight(index):
    with pytest.raises(ValueError):
        HybridRetriever(_lexical(), VectorRetriever(index, FakeEmbedder()), weight=1.5)


def test_hybrid_survives_single_side_empty(index):
    """向量侧失败时，融合层必须还能返回词法结果，而不是整体空。"""
    hybrid = HybridRetriever(_lexical(), VectorRetriever(index, ExplodingEmbedder()), weight=0.5)
    hits = hybrid.search_codes("眼科超声诊断设备", limit=3)
    assert hits and hits[0][0] == "eye"


# ---------------------------------------------------------------------------
# 语料构建
# ---------------------------------------------------------------------------


def test_build_corpus_texts_truncates_and_skips_empty():
    codes, texts = build_corpus_texts(
        [("a", "长文本" * 500), ("b", "   "), ("c", "短")], max_chars=20
    )
    assert codes == ["a", "c"]
    assert len(texts[0]) == 20
    assert texts[1] == "短"
