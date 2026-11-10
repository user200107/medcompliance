"""分类目录导入。

流程固定为三步，缺一步就会把脏数据带进判定引擎：

    读取 CSV → 结构校验（树完整性） → 只有零问题时才允许建树/写库

校验失败**不抛异常也不静默通过**，而是返回带问题清单的报告，
由调用方决定是中止还是人工修。导入是低频操作，宁可挡住也不将错就错。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mc_domain.enums import ManagementClass
from mc_domain.tree import ClassificationTree, CodeNode

#: CSV 的列名。顺序无所谓，但列名必须一致（防止把模板改坏了还不知道）。
REQUIRED_COLUMNS = ("code", "name", "parent_code")


@dataclass
class LoadReport:
    nodes: list[CodeNode] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    rows_read: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_if_invalid(self) -> None:
        if self.problems:
            preview = "\n".join(f"  - {p}" for p in self.problems[:20])
            more = f"\n  …… 另有 {len(self.problems) - 20} 条" if len(self.problems) > 20 else ""
            raise ValueError(f"分类目录校验未通过（{len(self.problems)} 项）：\n{preview}{more}")


def _parse_management_class(raw: str) -> ManagementClass | None:
    text = (raw or "").strip().upper()
    if not text:
        return None
    # 兼容「III类」「第三类」等写法
    mapping = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "一": "I", "二": "II", "三": "III"}
    text = text.replace("类", "").strip()
    text = mapping.get(text, text)
    try:
        return ManagementClass(text)
    except ValueError:
        return None


def load_csv(path: str | Path, *, catalog_version: str = "sample") -> LoadReport:
    """读取目录 CSV。utf-8-sig 兼容 Excel 导出的 BOM。"""
    path = Path(path)
    report = LoadReport()
    if not path.exists():
        report.problems.append(f"文件不存在：{path}")
        return report

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            report.problems.append(f"缺少必需列：{', '.join(missing)}（实际列：{', '.join(header)}）")
            return report

        for line_no, row in enumerate(reader, start=2):
            report.rows_read += 1
            code = (row.get("code") or "").strip()
            name = (row.get("name") or "").strip()
            if not code or not name:
                report.problems.append(f"第 {line_no} 行：code 或 name 为空")
                continue

            raw_parent = (row.get("parent_code") or "").strip()
            parent_code = raw_parent or None

            # level 由父链推导，不接受外部输入——避免 CSV 里写错层级还蒙混过关
            level = 0 if parent_code is None else _depth_of(parent_code) + 1

            examples = [
                e.strip()
                for e in (row.get("examples") or "").replace("；", "|").replace("、", "|").split("|")
                if e.strip()
            ]

            report.nodes.append(
                CodeNode(
                    code=code,
                    name=name,
                    level=level,
                    parent_code=parent_code,
                    management_class=_parse_management_class(row.get("management_class") or ""),
                    definition=(row.get("definition") or "").strip() or None,
                    scope_note=(row.get("scope_note") or "").strip() or None,
                    exclusion_note=(row.get("exclusion_note") or "").strip() or None,
                    examples=examples,
                    catalog_version=catalog_version,
                    sort_order=int((row.get("sort_order") or 0) or 0),
                )
            )

    return report


def _depth_of(code: str) -> int:
    """「07-01-01」→ 2。编码层级即深度，不读额外字段。"""
    return code.count("-")


def build_tree(nodes: Sequence[CodeNode], version: str) -> ClassificationTree:
    """建树并跑完整性校验；不通过直接抛，避免脏树进入判定。"""
    seen: set[str] = set()
    for node in nodes:
        if node.code in seen:
            raise ValueError(f"分类目录中编码重复：{node.code}")
        seen.add(node.code)

    tree = ClassificationTree.from_nodes(nodes, version)
    problems = tree.validate()
    if problems:
        preview = "\n".join(f"  - {p}" for p in problems[:20])
        raise ValueError(f"分类目录树不完整（{len(problems)} 项）：\n{preview}")
    return tree


def sorted_for_insert(nodes: Sequence[CodeNode]) -> list[CodeNode]:
    """按 level 升序返回。

    必须先插父后插子，否则拿不到父节点的自增 id——这是树形导入最容易踩的坑。
    """
    return sorted(nodes, key=lambda n: (n.level, n.code))


def to_upsert_rows(
    nodes: Iterable[CodeNode], code_to_id: dict[str, int] | None = None
) -> list[dict]:
    """转成可直接喂给 ORM `insert(...).values(**row)` 的字典。

    两件事需要注意：
    1. 幂等键是 (catalog_version, code)——目录调整时同一编码可以有多版，
       不能只用 code 做唯一键。
    2. `parent_id` 无法从 CSV 直接得到，必须由调用方先插入父层、拿回 id 填进
       `code_to_id`，再调用本函数。传 None 时返回的行不含 parent_id，
       适用于「先建全部节点再回填父子关系」的两阶段写法。
    """
    rows: list[dict] = []
    for n in nodes:
        row = {
            "catalog_version": n.catalog_version,
            "code": n.code,
            "level": n.level,
            "name": n.name,
            "management_class": n.management_class,
            "definition": n.definition,
            "scope_note": n.scope_note,
            "exclusion_note": n.exclusion_note,
            "examples": n.examples,
            "sort_order": n.sort_order,
            "is_active": n.is_active,
            "path": n.path or n.code,
            "name_path": n.name_path or n.name,
        }
        if code_to_id is not None:
            row["parent_id"] = code_to_id.get(n.parent_code) if n.parent_code else None
        rows.append(row)
    return rows


__all__ = [
    "LoadReport",
    "load_csv",
    "build_tree",
    "sorted_for_insert",
    "to_upsert_rows",
    "REQUIRED_COLUMNS",
]
