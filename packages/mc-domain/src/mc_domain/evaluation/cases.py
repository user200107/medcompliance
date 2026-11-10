"""评测用例：从官方分类界定结果构造 (输入, 标签) 对。

一条界定记录就是一条天然用例：
    输入 = 产品名称 + 产品详情（企业自己写的组成与预期用途）
    标签 = 产品类别（官方裁决）

**为什么这批数据适合做评测集**
- 它不在任何分类目录里，不存在 train/test 泄漏（对照：目录的「品名举例」不能当评测输入，
  因为检索索引本身就包含 examples，等于背答案）。
- 送界定意味着归类本身有争议，所以它天然是**难例**，比随机抽样有信息量。
- 官方裁决，不需要自己标。

**合规**：原始数据只本地使用，不提交仓库。仓库里只保留 `fingerprint()` 的输出
（记录数、SHA-256、时间范围）作为「确实跑过」的凭据。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from mc_domain.evaluation.labels import (
    ExpectedOutcome,
    LabelDistribution,
    build_distribution,
    normalize_label,
)
from mc_domain.schemas import ProductProfile


@dataclass(frozen=True)
class AdjudicationCase:
    id: str
    product_name: str
    detail: str
    raw_label: str
    outcome: ExpectedOutcome
    notice_title: str
    publish_date: str

    @property
    def input_text(self) -> str:
        """给流水线的原始文本。名称放前面——它通常是信息密度最高的部分。"""
        return f"{self.product_name}。{self.detail}".strip("。")

    def to_profile(self) -> ProductProfile:
        """绕过 LLM 抽取，直接把官方描述当作已知画像喂给判定阶段。

        评测分类引擎时这么做才公平：否则测的是"抽取器 + 判定器"两件事叠加，
        出错时无法定位是哪一层的问题。抽取器单独评（另有指标）。
        """
        return ProductProfile(
            name=self.product_name,
            intended_use=self.detail,
            raw_text=self.input_text,
        )


def load_cases(path: str | Path) -> list[AdjudicationCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测数据不存在：{path}（先跑 scripts/crawl_adjudications.py）")

    cases: list[AdjudicationCase] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                AdjudicationCase(
                    id=row["id"],
                    product_name=row.get("product_name", ""),
                    detail=row.get("detail", ""),
                    raw_label=row.get("product_category", ""),
                    outcome=normalize_label(row.get("product_category", "")),
                    notice_title=row.get("notice_title", ""),
                    publish_date=row.get("publish_date", ""),
                )
            )
    return cases


def distribution(cases: list[AdjudicationCase]) -> LabelDistribution:
    return build_distribution([c.raw_label for c in cases])


def stratified_sample(
    cases: list[AdjudicationCase], n: int, *, seed: int = 20261110
) -> list[AdjudicationCase]:
    """按标签分层抽样。

    类别不平衡时直接随机抽会得到一个几乎单类的集合，指标随之失去意义。
    分层后每个类别按同一比例抽取，并保证每类至少 1 条。

    小样本类别会被全量纳入，这会让实际样本数略大于 n——这是有意的取舍：
    宁可多几条，也不要在报告里出现"某类 0 样本"。
    """
    groups: dict[ExpectedOutcome, list[AdjudicationCase]] = {}
    for case in cases:
        groups.setdefault(case.outcome, []).append(case)

    rng = random.Random(seed)
    total = len(cases) or 1
    picked: list[AdjudicationCase] = []
    for outcome, group in sorted(groups.items(), key=lambda kv: kv[0].value):
        quota = max(1, round(n * len(group) / total))
        quota = min(quota, len(group))
        rng.shuffle(group)
        picked.extend(group[:quota])

    rng.shuffle(picked)
    return picked


def fingerprint(path: str | Path) -> dict[str, object]:
    """数据集指纹：可提交，不含原文。用来自证「指标是在这批数据上跑的」。"""
    path = Path(path)
    digest = hashlib.sha256()
    count = 0
    dates: list[str] = []
    titles: set[str] = set()
    with path.open("rb") as fh:
        for line in fh:
            digest.update(line)
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("publish_date"):
                dates.append(row["publish_date"])
            if row.get("notice_title"):
                titles.add(row["notice_title"])
    return {
        "source": "中国食品药品检定研究院 · 医疗器械分类界定结果汇总",
        "source_url": "https://app.nidc.org.cn/biaogzx/dataGk.do?formAction=listFljdResult",
        "usage": "仅本地评测，未再分发；仓库不包含原始数据",
        "records": count,
        "sha256": digest.hexdigest(),
        "date_range": [min(dates), max(dates)] if dates else None,
        "notice_batches": len(titles),
    }


__all__ = [
    "AdjudicationCase",
    "distribution",
    "fingerprint",
    "load_cases",
    "stratified_sample",
]
