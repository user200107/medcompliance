"""分类目录树：结构 + 完整性校验 + 排除法匹配。

刻意**不依赖 SQLAlchemy 与数据库**：这里全是纯数据结构和纯函数，
可以在单测里直接喂 dict 构造树。ORM 实体通过 `from_orm()` 适配进来。

这么做的原因：分类判定是整个产品最需要反复回归的部分，
如果它绑死在 session 上，每次改规则都要起数据库才能测，最后就没人测了。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from mc_domain.enums import ManagementClass

# 排除说明里的引导词，剥离后剩下的才是可匹配的类别词
_EXCLUSION_PREFIXES = ("本类不包括", "本子目录不包括", "不包括", "不包含", "不用于", "除")
_SPLIT_RE = re.compile(r"[，,；;、。：:（）()]|以及|和|及|或")

#: 虚指词不能当排除依据——「本类不包括其他」若匹配「其他」会把一切都排除掉。
_STOP_TERMS = frozenset({"其他", "其它", "等", "上述", "如上", "另行", "另外"})


@dataclass
class CodeNode:
    """分类目录里的一个节点（纯数据）。"""

    code: str
    name: str
    level: int
    parent_code: str | None = None
    management_class: ManagementClass | None = None
    definition: str | None = None
    scope_note: str | None = None
    exclusion_note: str | None = None
    examples: list[str] = field(default_factory=list)
    catalog_version: str = "2017"
    is_active: bool = True
    sort_order: int = 0

    # 建树时填充
    path: str = ""
    name_path: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.management_class, str):
            self.management_class = ManagementClass(self.management_class)

    @property
    def search_blob(self) -> str:
        """给 BM25 / 向量索引用的一段文本。name_path 放最前，它承载层级语义。"""
        parts = [self.name_path or self.name]
        if self.definition:
            parts.append(self.definition)
        if self.scope_note:
            parts.append(self.scope_note)
        if self.examples:
            parts.append("、".join(self.examples))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# 树
# ---------------------------------------------------------------------------


class ClassificationTree:
    """内存中的分类目录。整个目录只有几百个节点，全量加载最省事也最快。"""

    def __init__(self, nodes: dict[str, CodeNode], version: str) -> None:
        self._nodes = nodes
        self.version = version
        self._children: dict[str | None, list[str]] = {}
        for node in nodes.values():
            self._children.setdefault(node.parent_code, []).append(node.code)
        for siblings in self._children.values():
            siblings.sort(key=lambda c: self._nodes[c].sort_order or 0)
        self._recompute_paths()

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_nodes(cls, nodes: Iterable[CodeNode], version: str | None = None) -> ClassificationTree:
        index: dict[str, CodeNode] = {}
        for node in nodes:
            if node.code in index:
                raise ValueError(f"重复的分类编码: {node.code}")
            index[node.code] = node
        if version is None:
            version = next(iter(index.values())).catalog_version if index else "unknown"
        return cls(index, version)

    @classmethod
    def from_orm(cls, rows: Iterable[object], version: str | None = None) -> ClassificationTree:
        """适配 SQLAlchemy 实体（duck-typing，避免此处 import ORM）。"""
        nodes = [
            CodeNode(
                code=r.code,  # type: ignore[attr-defined]
                name=r.name,  # type: ignore[attr-defined]
                level=r.level,  # type: ignore[attr-defined]
                parent_code=(r.parent.code if getattr(r, "parent", None) else None),
                management_class=r.management_class,  # type: ignore[attr-defined]
                definition=r.definition,  # type: ignore[attr-defined]
                scope_note=r.scope_note,  # type: ignore[attr-defined]
                exclusion_note=r.exclusion_note,  # type: ignore[attr-defined]
                examples=list(r.examples or []),  # type: ignore[attr-defined]
                catalog_version=r.catalog_version,  # type: ignore[attr-defined]
                is_active=r.is_active,  # type: ignore[attr-defined]
                sort_order=r.sort_order,  # type: ignore[attr-defined]
            )
            for r in rows
        ]
        return cls.from_nodes(nodes, version)

    def _recompute_paths(self) -> None:
        for node in self._nodes.values():
            chain = self._chain_to_root(node)
            node.path = "/".join(n.code for n in chain)
            node.name_path = "/".join(n.name for n in chain)

    def _chain_to_root(self, node: CodeNode) -> list[CodeNode]:
        chain = [node]
        seen = {node.code}
        cursor = node
        while cursor.parent_code:
            parent = self._nodes.get(cursor.parent_code)
            if parent is None or parent.code in seen:  # 断链或成环
                break
            seen.add(parent.code)
            chain.append(parent)
            cursor = parent
        return list(reversed(chain))

    # -- 查询 ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, code: object) -> bool:
        return code in self._nodes

    def get(self, code: str) -> CodeNode:
        return self._nodes[code]

    def maybe_get(self, code: str) -> CodeNode | None:
        return self._nodes.get(code)

    def __iter__(self) -> Iterator[CodeNode]:
        return iter(self._nodes.values())

    def roots(self) -> list[CodeNode]:
        """所有子目录节点（level 0）。"""
        return [self._nodes[c] for c in self._children.get(None, [])]

    def children_of(self, code: str) -> list[CodeNode]:
        return [self._nodes[c] for c in self._children.get(code, [])]

    def leaves(self) -> list[CodeNode]:
        """可定类的末端节点。"""
        return [n for n in self._nodes.values() if n.code not in self._children]

    def ancestors(self, code: str) -> list[CodeNode]:
        """不含自身的祖先链，从根到父。"""
        node = self._nodes[code]
        return self._chain_to_root(node)[:-1]

    def subtree(self, code: str) -> list[CodeNode]:
        """含自身。"""
        out: list[CodeNode] = []
        stack = [code]
        while stack:
            cur = stack.pop()
            node = self._nodes.get(cur)
            if node is None:
                continue
            out.append(node)
            stack.extend(self._children.get(cur, []))
        return out

    def is_within(self, code: str, ancestor_code: str) -> bool:
        node = self._nodes.get(code)
        return bool(node and (node.path == ancestor_code or node.path.startswith(ancestor_code + "/")))

    def search_corpus(
        self, *, aggregate_containers: bool = False, max_examples_per_child: int = 3
    ) -> list[tuple[str, str]]:
        """(code, 检索文本) 列表，用于一次性构建索引。

        `aggregate_containers=True` 时，**容器节点**（有子节点的）的文本会拼上整个
        子树的名字与举例。不开这个开关会踩一个很隐蔽的坑：

            子目录节点的文本只有名字，如「临床检验器械」（8 个字）。
            查询「基因测序用测序试剂盒」与它的字符 bigram 重叠是 **0**，
            于是粗筛阶段一条都召不回，整条链路直接 no_match。

        实测 53 条样本里 8 条 no_match，**全部**卡在粗筛，且都是这类长描述产品。
        聚合后子目录带上了「基因测序」「试剂盒」等子节点词汇，粗筛才能命中。
        """
        out: list[tuple[str, str]] = []
        for node in self._nodes.values():
            blob = node.search_blob
            if aggregate_containers and node.code in self._children:
                extras: list[str] = []
                for descendant in self.subtree(node.code):
                    if descendant.code == node.code:
                        continue
                    extras.append(descendant.name)
                    if descendant.examples:
                        extras.extend(descendant.examples[:max_examples_per_child])
                if extras:
                    # 截断避免超长节点把索引撑爆；粗筛只需要能命中
                    blob = (blob + " " + " ".join(extras))[:2000]
            if blob:
                out.append((node.code, blob))
        return out

    # -- 完整性校验 ---------------------------------------------------------

    def validate(self) -> list[str]:
        """返回问题列表，空表示目录完整。导入目录后**必须**跑一遍。"""
        problems: list[str] = []

        for node in self._nodes.values():
            if node.parent_code is None:
                if node.level != 0:
                    problems.append(f"{node.code}: 根节点 level 应为 0，实际 {node.level}")
            else:
                parent = self._nodes.get(node.parent_code)
                if parent is None:
                    problems.append(f"{node.code}: 父节点 {node.parent_code} 不存在")
                elif node.level != parent.level + 1:
                    problems.append(
                        f"{node.code}: level {node.level} 与父节点 {parent.code} "
                        f"(level {parent.level}) 不连续"
                    )

            expected_path = "/".join(n.code for n in self._chain_to_root(node))
            if node.path and node.path != expected_path:
                problems.append(f"{node.code}: path 与本树推导不一致（{node.path} != {expected_path}）")

            if node.parent_code and not node.code.startswith(node.parent_code + "-"):
                problems.append(f"{node.code}: 编码前缀与父节点 {node.parent_code} 不匹配")

            has_children = node.code in self._children
            if not has_children and node.management_class is None:
                problems.append(f"{node.code}: 叶子节点缺少管理类别，将无法输出结论")

        # 环检测
        for node in self._nodes.values():
            chain = self._chain_to_root(node)
            if len(chain) != len({n.code for n in chain}):
                problems.append(f"{node.code}: 祖先链存在环")

        return problems


# ---------------------------------------------------------------------------
# 排除法
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExclusionHit:
    code: str
    matched_term: str
    note: str

    def as_evidence_text(self) -> str:
        return f"「{self.matched_term}」命中 {self.code} 的排除说明：{self.note}"


def extract_exclusion_terms(note: str) -> list[str]:
    """从「本类不包括：单纯消毒设备、口腔科用设备」里抽出可匹配的类别词。

    规则简单但可解释：剥引导词 → 按标点/连词切 → 去空白 → 丢掉过短片段。
    故意不做分词：分词会引入不确定性，而这一步必须完全确定、可复现。
    """
    if not note:
        return []
    text = note.strip()
    for prefix in _EXCLUSION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = [p.strip(" 　。；;：:") for p in _SPLIT_RE.split(text)]
    return [p for p in parts if len(p) >= 2 and p not in _STOP_TERMS]


class ExclusionMatcher:
    """确定性排除匹配器。product 文本里出现排除词即判该类别不适用。

    注意这是**保守**策略：宁可多排除（进 rejected 留痕），也不让一个明显属于
    其他子目录的产品混进候选。误排除会在 rejected 里可见，用户能推翻。
    """

    def __init__(self, normalizer: Callable[[str], str] | None = None) -> None:
        self._normalize = normalizer or _default_normalize

    def match(self, node: CodeNode, product_text: str) -> ExclusionHit | None:
        if not node.exclusion_note:
            return None
        haystack = self._normalize(product_text)
        for term in extract_exclusion_terms(node.exclusion_note):
            if self._normalize(term) in haystack:
                return ExclusionHit(code=node.code, matched_term=term, note=node.exclusion_note)
        return None


_FULLWIDTH = str.maketrans({"（": "(", "）": ")", "　": " "})


def _default_normalize(text: str) -> str:
    """全角转半角 + 小写 + 去空白。不做同义词扩展——那是检索层的活。"""
    return text.translate(_FULLWIDTH).lower().replace(" ", "").replace("\n", "")


__all__ = [
    "CodeNode",
    "ClassificationTree",
    "ExclusionHit",
    "ExclusionMatcher",
    "extract_exclusion_terms",
]
