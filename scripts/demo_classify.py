"""端到端演示：把一个产品描述跑完分类流水线，打印完整依据链。

    python scripts/demo_classify.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for pkg in (ROOT / "packages").glob("*/src"):
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

from mc_domain import (  # noqa: E402
    ClassificationPipeline,
    KeywordCodeRetriever,
    ProductProfile,
    build_tree,
    load_csv,
)
from mc_domain.enums import ContactType, EnergyType  # noqa: E402


class StaticNormalizer:
    """演示用：真实实现会调 LLM 做槽位抽取，这里直接给定画像。

    注意即使换成真实 LLM，它也只负责把口语翻译成 ProductProfile，
    分类判定仍然走下面的确定性阶段。
    """

    def __init__(self, profile: ProductProfile) -> None:
        self._profile = profile

    async def normalize(self, raw_text: str) -> ProductProfile:
        return self._profile


async def main() -> None:
    csv_path = (
        ROOT / "packages" / "mc-domain" / "src" / "mc_domain" / "data" / "catalog_sample.csv"
    )
    report = load_csv(csv_path, catalog_version="sample-synthetic")
    report.raise_if_invalid()
    tree = build_tree(report.nodes, "sample-synthetic")

    retriever = KeywordCodeRetriever(tree.search_corpus())
    profile = ProductProfile(
        name="超声治疗设备",
        intended_use="用于缓解肌肉疼痛，促进局部血液循环",
        contact_type=ContactType.SURFACE,
        energy_type=EnergyType.ACTIVE,
    )
    pipeline = ClassificationPipeline(tree, retriever, StaticNormalizer(profile))
    result = await pipeline.classify("一台用于缓解肌肉疼痛的超声治疗设备", profile=profile)

    print(f"状态：{result.status}    引擎：{result.engine}    耗时：{result.elapsed_ms} ms")
    print(f"依据日期：{result.as_of}")
    print()

    print("── 判定路径 ──")
    for i, step in enumerate(result.steps, 1):
        rule = f"  [{step.rule_key}]" if step.rule_key else ""
        print(f"{i}. {step.stage:<10} {step.summary}{rule}")
        if step.kept or step.dropped:
            print(f"   保留 {step.kept} / 剔除 {step.dropped}")
    print()

    if result.best:
        print("── 结论 ──")
        print(f"分类编码：{result.best.code}   管理类别：{result.best.management_class.value} 类")
        print(f"类别路径：{result.best.name_path}")
        print("依据：")
        for ev in result.best.evidence:
            print(f"  · ({ev.kind}) {ev.text}")
        print()

    if result.candidates:
        print("── Top-{k} 候选 ──".format(k=len(result.candidates)))
        for rank, cand in enumerate(result.candidates, 1):
            cls = cand.management_class.value if cand.management_class else "-"
            print(f"  {rank}. {cand.code}  {cand.name}  ({cls} 类, score={cand.score})")
        print()

    if result.rejected:
        print("── 已排除 ──")
        for item in result.rejected:
            print(f"  × {item.code} {item.name}：{item.reason}")
            print(f"     依据：{item.evidence.text}")
        print()

    if result.missing_slots:
        print("── 需要补充 ──")
        for slot in result.missing_slots:
            print(f"  ? {slot.question}")
            if slot.options:
                print(f"    可选项：{' / '.join(slot.options)}")
        print()

    print("── 自检提示 ──")
    for caveat in result.caveats:
        print(f"  ! {caveat}")


if __name__ == "__main__":
    asyncio.run(main())
