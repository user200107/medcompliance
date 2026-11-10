"""分类目录加载与运行时对象。

目录只在启动时读一次：全量几百个节点，常驻内存最省事，也避免了每次请求查库。
真实目录从 `MC_CATALOG_CSV` 指向的 CSV 载入（结构见 mc_domain.catalog_loader）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from mc_domain import (
    ClassificationTree,
    KeywordCodeRetriever,
    build_tree,
    load_csv,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogRuntime:
    tree: ClassificationTree
    retriever: KeywordCodeRetriever
    version: str

    @property
    def node_count(self) -> int:
        return len(self.tree)

    @property
    def leaf_count(self) -> int:
        return len(self.tree.leaves())

    @property
    def classifiable_count(self) -> int:
        """能给出管理类别结论的节点数。与 leaves 通常相等，但语义不同：
        目录里允许存在带管理类别但仍有子节点的层级。"""
        return sum(1 for node in self.tree if node.management_class is not None)


def load_catalog(csv_path: str | Path, version: str | None = None) -> CatalogRuntime:
    """载入并校验目录。校验不通过直接抛，不让脏目录进入判定。"""
    report = load_csv(csv_path, catalog_version=version or "unknown")
    report.raise_if_invalid()

    effective_version = version or (report.nodes[0].catalog_version if report.nodes else "unknown")
    tree = build_tree(report.nodes, effective_version)

    runtime = CatalogRuntime(
        tree=tree,
        # aggregate_containers=True：子目录节点的检索文本必须带上子树词汇，
        # 否则长产品描述在粗筛阶段与「临床检验器械」这类 8 字名字零重叠，
        # 会直接 no_match（实测 8/53）。
        retriever=KeywordCodeRetriever(tree.search_corpus(aggregate_containers=True)),
        version=effective_version,
    )
    logger.info(
        "分类目录载入完成：version=%s 节点=%d 可定类=%d",
        runtime.version,
        runtime.node_count,
        runtime.leaf_count,
    )
    return runtime
