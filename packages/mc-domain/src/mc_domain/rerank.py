"""重排序：两段式检索的第二段。

## 为什么值得接

端到端评测暴露了一个特征：**Top-3 98.9%、Top-1 62.3%**。
正确类别几乎总在候选里，问题在排序。这正是重排序模型的用武之地——
它对 (query, 候选) 逐条做交叉注意力打分，理论上限就是那个 98.9%。

## 实测结论：**P1 默认不启用重排序**

在 90 条官方界定样本上（qwen3-rerank，α=0 只看排序质量）：

| 路径 | Top-1 | Ⅰ召回 | Ⅲ召回 |
|---|---|---|---|
| 单段（字符 bigram） | **68.9%** | 79.2% | 56.0% |
| 纯重排（compact） | 64.4% | 45.8% | 52.0% |
| 纯重排（full） | 63.3% | 41.7% | 68.0% |

两条观察：
1. **Top-3 不动（96.7%）**——符合预期，重排序不改变召回。
2. **Top-1 变差，且 Ⅰ 类召回塌陷、Ⅲ 类上升**——它做的其实是**类别偏置**。

而"往高类别偏"这件事，`decision.py` 的 `risk_averse` / `max_recall_iii`
已经能确定性做到，**不花钱、可复现、可解释**。所以重排序在本任务上是重复投入。

原因也说得通：重排序优化的是**语义相关度**，而我们的目标是**监管类别**。
类别是命中条目的属性，不是文本相似度的函数；两者不是同一个目标函数。

客户端保留下来是因为 P2（标签审核）里重排序面对的是「条款 vs 标签项」的匹配，
那个任务的目标函数与语义相关度是对齐的。

## 接口差异（踩过一次的坑）

抽取和 embedding 都能走 OpenAI 兼容层，**重排序不行**。
DashScope 的 rerank 是独立原生接口：

    POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
    {"model": "...", "input": {"query": "...", "documents": [...]},
     "parameters": {"return_documents": false, "top_n": 3}}

响应是 `{"output": {"results": [{"index": 0, "relevance_score": 0.72}, ...]}}`，
**index 是原文档下标**，所以要自己映射回 code。

## 失败语义

重排序是**增强**，不是必需项。它挂了不能让主流程崩：
降级回第一段的顺序，但必须留痕（`failures` 计数 + `last_error`），
并在日志里 warn。静默降级等于假装有重排序——那是上一代项目踩过的坑。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import httpx

from mc_domain.env import http_client_kwargs
from mc_domain.retrieval import KeywordCodeRetriever

logger = logging.getLogger(__name__)


class RerankFailed(RuntimeError):
    """重排序调用失败。调用方应降级并留痕，**不得**当作成功。"""


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """返回 [(文档下标, 相关度)]，按相关度降序。下标对应传入的 documents。"""
        ...


class NullReranker:
    """不做事。保持接口一致，便于关闭重排序而不改调用方。"""

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        return [(i, 1.0 - i * 1e-6) for i in range(len(documents))]


class DashScopeReranker:
    """DashScope 原生 rerank 客户端。

    单条文档上限 4000 token（qwen3-rerank），这里按字符截断兜底，
    避免某个节点的品名举例特别长时整批失败。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: int = 30,
        max_retries: int = 2,
        max_doc_chars: int = 1000,
    ) -> None:
        if not base_url:
            raise ValueError("rerank 需要 base_url")
        if not api_key:
            raise ValueError("rerank 需要 api_key")
        self._url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._max_doc_chars = max_doc_chars
        self.last_usage: dict[str, int] | None = None

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        payload: dict[str, object] = {
            "model": self._model,
            "input": {
                "query": query,
                "documents": [d[: self._max_doc_chars] for d in documents],
            },
            "parameters": {"return_documents": False},
        }
        if top_n is not None:
            payload["parameters"] = {**payload["parameters"], "top_n": int(top_n)}  # type: ignore[dict-item]

        last_error: Exception | None = None
        with httpx.Client(**http_client_kwargs(self._timeout)) as client:  # type: ignore[arg-type]
            for attempt in range(self._max_retries + 1):
                try:
                    resp = client.post(
                        self._url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    self.last_usage = body.get("usage")
                    results = body["output"]["results"]
                    out = [
                        (int(item["index"]), float(item["relevance_score"]))
                        for item in results
                    ]
                    out.sort(key=lambda pair: pair[1], reverse=True)
                    return out
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning("重排序第 %d 次失败：%s", attempt + 1, exc)

        raise RerankFailed(f"重排序失败（已重试 {self._max_retries} 次）：{last_error}")


class TwoStageRetriever:
    """第一段召回 N 条 → 第二段重排序 → 返回前 limit 条。

    与 `KeywordCodeRetriever` 的接口完全一致，所以可以原地替换：
    流水线、评测脚本都不用改。
    """

    def __init__(
        self,
        base: KeywordCodeRetriever,
        reranker: Reranker,
        code_text: dict[str, str],
        *,
        recall_k: int = 20,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.code_text = code_text
        self.recall_k = recall_k
        self.failures = 0
        self.last_error: str | None = None

    def search_codes(
        self, query: str, *, limit: int, allowed_codes: set[str] | None = None
    ) -> list[tuple[str, float]]:
        hits = self.base.search_codes(query, limit=self.recall_k, allowed_codes=allowed_codes)
        if not hits:
            return []

        documents = [self.code_text.get(code, code) for code, _ in hits]
        try:
            order = self.reranker.rerank(query, documents, top_n=min(limit, len(documents)))
        except RerankFailed as exc:
            # 增强层失败：降级回第一段顺序，但必须留痕
            self.failures += 1
            self.last_error = str(exc)
            logger.warning("重排序失败，降级为单段检索：%s", exc)
            return hits[:limit]

        ranked = [(hits[idx][0], score) for idx, score in order if 0 <= idx < len(hits)]
        if not ranked:  # 服务端返回空结果也视为退化
            self.failures += 1
            self.last_error = "重排序返回空结果"
            return hits[:limit]
        return ranked[:limit]


def build_doc_text(node, mode: str = "compact") -> str:
    """给重排序用的文档文本。

    - `compact`：类别路径 + 预期用途 + 少量举例。省 token，突出语义要点。
    - `full`：在 compact 基础上补上类别定义与**全部**品名举例。

    两种模式是有意保留的：实测发现本任务的信号里「品名举例字面命中」占比很高，
    而 compact 只取 5 条举例会把这一路信号砍掉。要不要开 full 取决于
    重排序是"帮你看语义"还是"帮你看字面"——这个差别必须能被测出来，
    所以别把模式写死。
    """
    parts = [node.name_path]
    if node.scope_note:
        parts.append(node.scope_note)
    if node.definition and (mode == "full" or not node.scope_note):
        parts.append(node.definition)
    if node.examples:
        examples = node.examples if mode == "full" else node.examples[:5]
        parts.append("举例：" + "、".join(examples))
    return "。".join(p.strip("。") for p in parts if p)


__all__ = [
    "DashScopeReranker",
    "NullReranker",
    "RerankFailed",
    "Reranker",
    "TwoStageRetriever",
    "build_doc_text",
]
