"""检索层的最小实现。

分类目录只有几百个节点，生产上会走 Milvus + BM25 混合检索（直接复用 fintech-brain 的
`rag/retriever.py`）。但这里需要一个**零依赖、完全确定**的实现，理由有三：

1. 单测不该连数据库和向量库，否则最后没人跑；
2. 检索服务挂了，分类要有降级路径（不能整个功能 500）；
3. 它是评测的**基线**——混合检索打不过字符 bigram 就说明向量没起作用。

实现是字符 bigram + IDF 加权。中文短文本上这个基线意外地强。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


def _bigrams(text: str) -> Counter[str]:
    """字符 bigram。剔除空白与标点，避免把格式差异当成语义差异。"""
    cleaned = "".join(ch for ch in text if ch.isalnum())
    if len(cleaned) < 2:
        return Counter({cleaned: 1}) if cleaned else Counter()
    return Counter(cleaned[i : i + 2] for i in range(len(cleaned) - 1))


class KeywordCodeRetriever:
    """基于 IDF 加权 bigram 重叠的检索器。

    打分 = 查询 bigram 与文档 bigram 的加权交集 / sqrt(查询 bigram 数)。
    除以 sqrt 是为了不让长查询（描述详细的产品）系统性占优。
    """

    def __init__(self, corpus: Iterable[tuple[str, str]]) -> None:
        self._docs: list[tuple[str, Counter[str]]] = []
        df: Counter[str] = Counter()
        for code, text in corpus:
            grams = _bigrams(text)
            if not grams:
                continue
            self._docs.append((code, grams))
            df.update(grams.keys())
        total = max(len(self._docs), 1)
        self._idf: dict[str, float] = {
            gram: math.log((total + 1) / (count + 1)) + 1.0 for gram, count in df.items()
        }

    def __len__(self) -> int:
        return len(self._docs)

    def search_codes(
        self, query: str, *, limit: int, allowed_codes: set[str] | None = None
    ) -> list[tuple[str, float]]:
        query_grams = _bigrams(query)
        if not query_grams:
            return []
        # 查询里未在任何文档出现过的 bigram 给最低 idf，仍参与分母
        unknown_idf = math.log(len(self._docs) + 2)
        norm = math.sqrt(sum(query_grams.values())) or 1.0

        scored: list[tuple[str, float]] = []
        for code, doc_grams in self._docs:
            if allowed_codes is not None and code not in allowed_codes:
                continue
            overlap = sum(
                min(count, doc_grams[gram]) * self._idf.get(gram, unknown_idf)
                for gram, count in query_grams.items()
                if gram in doc_grams
            )
            if overlap > 0:
                scored.append((code, overlap / norm))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


__all__ = ["KeywordCodeRetriever"]
