"""在真实分类界定数据上跑评测，产出报告。

    python scripts/eval_adjudications.py                      # 默认抽 400 条
    python scripts/eval_adjudications.py --n 800
    python scripts/eval_adjudications.py --all                # 全量（慢）
    python scripts/eval_adjudications.py --catalog path/to/real_catalog.csv

产出：
    docs/evaluation/baseline-<日期>.md      可提交（含指纹，不含原文）
    data/raw/last_eval.json                本地明细（已被 .gitignore 忽略）

设计要点：
- 抽的是**分层样本**，不是前 N 条。站内按发布时间正序，直接取前 N 条会得到
  一个「只有 2019 年」的偏样本。
- 门控与类别分开评：门控不依赖分类目录，现在就能给出真实数字；
  类别准确率则被"目录只有 8 个合成节点"卡住，报告里会写明这一点而不是硬报一个数。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in sorted((ROOT / "packages").glob("*/src")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from mc_domain.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mc_domain.catalog_loader import build_tree, load_csv  # noqa: E402
from mc_domain.classify import ClassificationPipeline  # noqa: E402
from mc_domain.enums import ClassificationOutcome  # noqa: E402
from mc_domain.evaluation import (  # noqa: E402
    AbstainStats,
    ExpectedOutcome,
    class_metrics,
    confusion_matrix,
    distribution,
    fingerprint,
    gate_metrics,
    load_cases,
    stratified_sample,
)
from mc_domain.evaluation.report import EvalReport  # noqa: E402
from mc_domain.retrieval import KeywordCodeRetriever  # noqa: E402
from mc_domain.schemas import ProductProfile  # noqa: E402

#: 复用 P1 的抽取器，保证「评测时」和「线上时」用的是同一套槽位逻辑
from mc_medireg.normalizer import NormalizationFailed, build_normalizer  # noqa: E402

logger = logging.getLogger("eval")

DEFAULT_DATA = ROOT / "data" / "raw" / "adjudications.jsonl"
DEFAULT_CATALOG = ROOT / "packages" / "mc-domain" / "src" / "mc_domain" / "data" / "catalog_sample.csv"
REPORT_DIR = ROOT / "docs" / "evaluation"
DETAIL_OUT = ROOT / "data" / "raw" / "last_eval.json"


async def run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"评测数据不存在：{data_path}", file=sys.stderr)
        print("先跑：python scripts/crawl_adjudications.py --all --sample-every 4", file=sys.stderr)
        return 2

    cases = load_cases(data_path)
    fp = fingerprint(data_path)
    dist = distribution(cases)
    print(f"载入 {len(cases)} 条，时间范围 {fp['date_range']}")
    print(dist.render())

    sample = cases if args.all else stratified_sample(cases, args.n)
    # 系统表达不了的结论（不单独作为器械 / 药械组合 / 视具体情况而定 / 未知）
    # 一律排除出指标——硬塞进分母只会让数字失真。
    sample = [c for c in sample if c.outcome.in_gate_scope]
    print(f"\n分层抽样：{len(sample)} 条")

    catalog_path = Path(args.catalog)
    report = load_csv(catalog_path, catalog_version="eval")
    report.raise_if_invalid()
    tree = build_tree(report.nodes, "eval")
    print(f"分类目录：{len(tree)} 个节点，其中可定类 {sum(1 for n in tree if n.management_class)} 个")

    normalizer = build_normalizer(
        args.normalizer,
        base_url=os.environ.get("MC_LLM_BASE_URL", ""),
        api_key=os.environ.get("MC_LLM_API_KEY", ""),
        model=os.environ.get("MC_LLM_MODEL", "qwen-plus"),
        timeout_s=int(os.environ.get("MC_LLM_TIMEOUT_S", "30") or 30),
        max_retries=int(os.environ.get("MC_LLM_MAX_RETRIES", "2") or 2),
    )
    print(f"槽位抽取器：{args.normalizer}")
    pipeline = ClassificationPipeline(
        tree=tree,
        retriever=KeywordCodeRetriever(tree.search_corpus(aggregate_containers=True)),
        normalizer=normalizer,
    )

    gate_pairs: list[tuple[bool | None, bool | None]] = []
    class_samples: list[tuple[ExpectedOutcome, list[ClassificationOutcome]]] = []
    confusion_pairs: list[tuple[ExpectedOutcome, ClassificationOutcome | None]] = []
    abstain = AbstainStats(total=len(sample))
    extraction_failures = 0
    class_expected = 0
    details: list[dict[str, object]] = []

    as_of = date.today()
    for i, case in enumerate(sample, 1):
        profile = await pipeline.normalizer.normalize(case.input_text)

        # 门控**独立**评测：它的策略是「没有确定性否定表述就默认视为器械」。
        # 这是实际运营策略（不可能把所有产品都拒答），代价是漏掉 not-device 类，
        # 必须把这个代价量化出来，而不是只统计"闸门有把握时"的那几条。
        gate_decision = pipeline.gate.evaluate(profile).decision
        pred_device_gate = gate_decision.verdict != "not_device"
        gate_pairs.append((case.outcome.is_device, pred_device_gate))

        result = await pipeline.classify(case.input_text, as_of=as_of, profile=profile)

        if result.status == "needs_more_info":
            abstain.abstained += 1
        elif result.status == "no_match":
            abstain.no_match += 1

        # 类别指标只在「系统确实给出了结论」的样本上统计。
        # 把拒答算成错答，会把"覆盖不足"和"判错"混成一个数字，
        # 而这两件事的改进路径完全不同。覆盖率另行报告。
        if result.status == "classified" and case.outcome.in_class_scope:
            preds = []
            if result.outcome is not None and result.outcome.is_device:
                preds.append(result.outcome)
            for cand in result.candidates:
                outcome = ClassificationOutcome.from_management_class(cand.management_class)
                if outcome is not None and outcome.is_device and outcome not in preds:
                    preds.append(outcome)
            class_samples.append((case.outcome, preds))
            confusion_pairs.append((case.outcome, result.outcome))
        elif case.outcome.in_class_scope:
            class_expected += 1

        details.append(
            {
                "id": case.id,
                "name": case.product_name,
                "gold": case.outcome.value,
                "raw_label": case.raw_label,
                "status": result.status,
                "pred": None if result.outcome is None else result.outcome.value,
                "gate": None if result.gate is None else result.gate.verdict,
                "steps": [s.stage for s in result.steps],
            }
        )
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}")

    gate = gate_metrics(gate_pairs)
    classes = class_metrics(class_samples)

    # 多数类基线：没有它，准确率无法解释
    gate_golds = [g for g, _ in gate_pairs]
    device_rate = sum(1 for g in gate_golds if g) / len(gate_golds) if gate_golds else 0.0
    baselines = {
        "多数类基线（把全部样本都判为器械）": device_rate,
    }

    notes = build_notes(
        tree=tree,
        catalog_path=catalog_path,
        dist=dist,
        abstain=abstain,
        gate=gate,
        classes=classes,
        class_expected=class_expected,
        device_baseline=device_rate,
    )

    out = EvalReport(
        fingerprint=fp,
        distribution=dist,
        gate=gate,
        classes=classes,
        abstain=abstain,
        sampled=len(sample),
        catalog={
            "path": str(catalog_path),
            "nodes": len(tree),
            "classifiable": sum(1 for n in tree if n.management_class is not None),
            "version": "eval",
        },
        confusion=confusion_matrix(confusion_pairs),
        baselines=baselines,
        notes=notes,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"baseline-{date.today():%Y%m%d}.md"
    md_path.write_text(out.to_markdown(), encoding="utf-8")
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_OUT.write_text(
        json.dumps({"report": out.to_dict(), "cases": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("门控指标")
    print(gate.render())
    print("\n类别指标")
    print(classes.render())
    print("\n" + abstain.render())
    if extraction_failures:
        print(f"  抽取失败 {extraction_failures} 条（已计入拒答）")
    calls = getattr(normalizer, "calls", 0)
    if calls:
        tokens = getattr(normalizer, "total_tokens", 0)
        print(f"  抽取调用 {calls} 次，累计 token {tokens}")
    print(f"\n报告：{md_path}")
    print(f"明细：{DETAIL_OUT}（本地，不提交）")
    return 0


def build_notes(
    *, tree, catalog_path, dist, abstain, gate, classes, class_expected, device_baseline
) -> list[str]:
    notes: list[str] = []
    classifiable = sum(1 for n in tree if n.management_class is not None)
    is_synthetic = classifiable < 50

    # ---- 门控 ----
    delta = gate.accuracy - device_baseline
    notes.append(
        f"**门控（规则版）不可用，这是本轮最重要的结论。** "
        f"策略是「没有确定性否定表述就默认视为器械」，于是召回率 100% 但特异度仅 "
        f"{gate.specificity:.1%}（TN={gate.tn} / FP={gate.fp}）。"
        f"准确率 {gate.accuracy:.1%} 看似不错，可多数类基线就有 {device_baseline:.1%}，"
        f"只高出 {delta * 100:+.1f} 个百分点——**这个闸门几乎没有提供信息**。"
        "它只能拦下表述里明写「无治疗作用 / 不涉及疾病」的少数样本。"
        "要真正解决，需要语义判定（模型 + 人工复核），而不是继续加关键词。"
    )

    # ---- 类别 ----
    if is_synthetic:
        notes.append(
            f"**类别指标没有解释力。** 分类目录只有 {len(tree)} 个节点"
            f"（可定类 {classifiable} 个），是合成数据。必须先导入真实目录。"
        )
    else:
        notes.append(
            f"**端到端类别指标被抽取层卡住，n 只有 {classes.n} 条**"
            f"（类别范围内共 {class_expected} 条）。目录已经是真实的"
            f"（{len(tree)} 个节点，可定类 {classifiable} 个），"
            "所以这个 0% 反映的不是判定能力，而是覆盖率不足。"
            "判定层本身的能力请用隔离实验衡量："
            "`python scripts/eval_retrieval.py`（绕过抽取，直接检索目录）。"
        )

    if abstain.coverage < 0.5:
        notes.append(
            f"覆盖率仅 {abstain.coverage:.1%}（转追问 {abstain.abstained} / 无匹配 {abstain.no_match}，"
            f"共 {abstain.total} 条）：瓶颈在**槽位抽取层**，不在目录、也不在判定层。"
            "官方描述是「由…组成，用于…」的写法，很少直接写接触类型与有源/无源。"
            "这量化了接入 LLM 抽取器的必要性——关键词基线覆盖不了这种文本。"
        )

    notes.append(
        "标签空间里有若干系统不产出的结论（不单独作为医疗器械、药械组合、"
        "视具体情况而定），已从指标中排除并单独计数。"
        "其中「视具体情况而定」是**官方自己不给唯一结论**——"
        "要求系统比监管机构更确定，是不公平的评测。"
    )

    notes.append(
        "数据来源为中国食品药品检定研究院《分类界定结果汇总》与《分类目录库》，"
        "仅本地评测使用，未再分发；仓库不包含原始数据，只保留指纹与聚合指标。"
    )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="在分类界定数据上评测")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--n", type=int, default=400, help="分层抽样条数")
    parser.add_argument("--all", action="store_true", help="全量评测（慢）")
    parser.add_argument("--include-unsupported", action="store_true", help="（预留）")
    parser.add_argument(
        "--normalizer",
        default="heuristic",
        choices=["heuristic", "llm"],
        help="槽位抽取器；llm 需要 MC_LLM_* 已配置（会产生 API 费用）",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
