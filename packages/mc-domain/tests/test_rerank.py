"""两段式检索的单测（全部离线，用假重排序器）。

要钉住的三件事：
1. 重排序能改变顺序，且 code 映射正确（不是靠下标巧合）；
2. 重排序失败时**降级但留痕**，不能静默装作成功；
3. `allowed_codes` 过滤在第一段就生效，重排序不能把被过滤掉的节点捞回来。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mc_domain.rerank import (
    NullReranker,
    RerankFailed,
    TwoStageRetriever,
    build_doc_text,
)
from mc_domain.retrieval import KeywordCodeRetriever
from mc_domain.tree import CodeNode


@dataclass
class FakeBase:
    hits: list[tuple[str, float]]
    last_allowed: set[str] | None = None

    def search_codes(self, query: str, *, limit: int, allowed_codes=None):
        self.last_allowed = allowed_codes
        out = self.hits
        if allowed_codes is not None:
            out = [(c, s) for c, s in out if c in allowed_codes]
        return out[:limit]


class ReversingReranker:
    """把顺序完全颠倒，便于断言"结果确实来自重排序"。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_docs: list[str] = []

    def rerank(self, query, documents, *, top_n=None):
        self.calls += 1
        self.last_docs = list(documents)
        n = len(documents)
        return [(n - 1 - i, 100.0 - (n - 1 - i)) for i in range(n)]


class ExplodingReranker:
    def rerank(self, query, documents, *, top_n=None):
        raise RerankFailed("模拟服务不可用")


def test_rerank_reorders_and_maps_codes():
    base = FakeBase([("a", 3.0), ("b", 2.0), ("c", 1.0)])
    reranker = ReversingReranker()
    retriever = TwoStageRetriever(base, reranker, {"a": "A", "b": "B", "c": "C"}, recall_k=10)

    out = retriever.search_codes("q", limit=3)
    assert [code for code, _ in out] == ["c", "b", "a"]
    assert reranker.calls == 1


def test_rerank_receives_doc_text_in_base_order():
    base = FakeBase([("a", 3.0), ("b", 2.0)])
    reranker = ReversingReranker()
    retriever = TwoStageRetriever(base, reranker, {"a": "文档A", "b": "文档B"}, recall_k=10)
    retriever.search_codes("q", limit=2)
    assert reranker.last_docs == ["文档A", "文档B"]


def test_failure_degrades_but_records():
    base = FakeBase([("a", 3.0), ("b", 2.0)])
    retriever = TwoStageRetriever(base, ExplodingReranker(), {"a": "A", "b": "B"})

    out = retriever.search_codes("q", limit=2)
    # 降级为第一段顺序
    assert [code for code, _ in out] == ["a", "b"]
    # 但必须留痕
    assert retriever.failures == 1
    assert retriever.last_error and "模拟服务不可用" in retriever.last_error


def test_allowed_codes_filter_applies_before_rerank():
    base = FakeBase([("a", 3.0), ("b", 2.0), ("c", 1.0)])
    reranker = ReversingReranker()
    retriever = TwoStageRetriever(base, reranker, {"a": "A", "b": "B", "c": "C"})

    out = retriever.search_codes("q", limit=5, allowed_codes={"a", "c"})
    assert {code for code, _ in out} == {"a", "c"}
    assert base.last_allowed == {"a", "c"}


def test_empty_base_skips_rerank():
    reranker = ReversingReranker()
    retriever = TwoStageRetriever(FakeBase([]), reranker, {})
    assert retriever.search_codes("q", limit=5) == []
    assert reranker.calls == 0


def test_null_reranker_preserves_order():
    base = FakeBase([("a", 3.0), ("b", 2.0)])
    retriever = TwoStageRetriever(base, NullReranker(), {"a": "A", "b": "B"})
    assert [c for c, _ in retriever.search_codes("q", limit=2)] == ["a", "b"]


def test_build_doc_text_prefers_intended_use():
    node = CodeNode(
        code="01-01",
        name="超声手术设备",
        level=2,
        management_class="III",
        definition="定义为长文本",
        scope_note="用于软组织的切割",
        examples=["超声刀", "超声切割止血刀"],
    )
    text = build_doc_text(node)
    assert "用于软组织的切割" in text
    assert "定义为长文本" not in text  # 预期用途优先，二者只取其一
    assert "超声刀" in text


def test_build_doc_text_falls_back_to_definition():
    node = CodeNode(code="01", name="X", level=0, definition="定义文本")
    assert "定义文本" in build_doc_text(node)


def test_keyword_retriever_interface_compatible():
    """两段式必须能原地替换单段检索——接口签名一致。"""
    tree_nodes = [
        CodeNode(code="a", name="超声治疗设备", level=0, management_class="III"),
        CodeNode(code="b", name="超声诊断设备", level=0, management_class="II"),
    ]
    base = KeywordCodeRetriever([(n.code, n.search_blob) for n in tree_nodes])
    two_stage = TwoStageRetriever(base, NullReranker(), {n.code: n.name for n in tree_nodes})
    for r in (base, two_stage):
        hits = r.search_codes("超声治疗设备", limit=2, allowed_codes={"a", "b"})
        assert isinstance(hits, list) and hits
