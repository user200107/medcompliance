"""给 P2 算指标：在合成标签集上评必标项引擎。

    python scripts/eval_labels.py --n 300
    python scripts/eval_labels.py --n 300 --mode text     # 连文本抽取一起测

## 两种模式

- `fields`：直接喂结构化字段 → 只测**判定引擎**。
- `text`：渲染成标签文本再走正则抽取 → 测**抽取 + 判定**整条链路。

分开跑是刻意的：出错时才能定位是"抽不出来"还是"判错了"。
（P1 的教训：抽取层和判定层混在一起测，最后定位不出来。）

## 指标

- **违规项 P/R/F1**（micro）：把每条样本的检出项集合与埋入项集合比对；
- **分缺陷类型召回**：哪一类缺陷最容易被漏；
- **负控假阳性率**：干净标签被误判成违规的比例——这个数字对用户体验最致命。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in sorted((ROOT / "packages").glob("*/src")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from mc_domain.enums import RequirementKey  # noqa: E402

from mc_labelguard.checklist import check_label  # noqa: E402
from mc_labelguard.extract import extract_from_text  # noqa: E402
from mc_labelguard.synthetic import (  # noqa: E402
    DEFECT_RENDERERS,
    SyntheticLabel,
    generate,
)


def evaluate(samples: list[SyntheticLabel], mode: str) -> dict:
    tp = fp = fn = 0
    clean_total = clean_fp = 0
    per_kind_total: Counter[str] = Counter()
    per_kind_hit: Counter[str] = Counter()
    undetermined_total = 0
    items_total = 0
    risk_match = 0

    for sample in samples:
        extraction = (
            sample.fields if mode == "fields" else extract_from_text(sample.text)
        )
        result = check_label(extraction)
        detected = set(result.violation_keys())
        planted = set(sample.planted)

        # 只在「应当检出」的项上统计，未植入的项不参与（避免把非强制项算进去）
        hit = detected & planted
        miss = planted - detected
        extra = {k for k in detected - planted}

        tp += len(hit)
        fn += len(miss)
        fp += len(extra)

        per_kind_total[sample.defect_kind] += 1
        if planted and not miss:
            per_kind_hit[sample.defect_kind] += 1

        if sample.clean:
            clean_total += 1
            if detected:
                clean_fp += 1
        else:
            expected_risk = "high"
            if result.risk_level.value == expected_risk:
                risk_match += 1

        undetermined_total += result.undeterminable
        items_total += result.total

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clean_total": clean_total,
        "clean_fp": clean_fp,
        "clean_fp_rate": clean_fp / clean_total if clean_total else 0.0,
        "per_kind": {
            k: (per_kind_hit[k], per_kind_total[k]) for k in sorted(per_kind_total)
        },
        "risk_high_rate": risk_match / (len(samples) - clean_total) if len(samples) > clean_total else 0.0,
        "undeterminable_rate": undetermined_total / items_total if items_total else 0.0,
    }


def print_report(title: str, m: dict, n: int) -> None:
    print(f"\n{title}（n={n}）")
    print("=" * 58)
    print(f"  违规项  P {m['precision']:.1%}   R {m['recall']:.1%}   F1 {m['f1']:.3f}")
    print(f"           TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(
        f"  负控假阳性 {m['clean_fp']}/{m['clean_total']} "
        f"= {m['clean_fp_rate']:.1%}  ← 干净标签被误判的比例"
    )
    print(f"  缺陷样本判为 high 风险的比例：{m['risk_high_rate']:.1%}")
    print(f"  不可判定项占比：{m['undeterminable_rate']:.1%}")
    print("\n  分缺陷类型召回：")
    for kind, (hit, total) in m["per_kind"].items():
        if kind == "clean":
            # 负控没有"应检出项"，出现在召回表里会显示成 0/50 的假漏判。
            # 它由上面的「负控假阳性」单独衡量。
            continue
        flag = "" if hit == total else "  ← 有漏判"
        print(f"    {DEFECT_RENDERERS.get(kind, kind):<22} {hit}/{total}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description="P2 必标项引擎评测")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument(
        "--mode", default="both", choices=["fields", "text", "both"], help="fields=只测判定，text=连抽取一起测"
    )
    args = parser.parse_args()

    samples = generate(args.n, seed=args.seed)
    print(f"合成标签 {len(samples)} 条（缺陷类型 {len(DEFECT_RENDERERS) - 1} 种 + 负控）")

    modes = ["fields", "text"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        m = evaluate(samples, mode)
        results[mode] = m
        label = "只测判定引擎（fields）" if mode == "fields" else "抽取 + 判定整链路（text）"
        print_report(label, m, len(samples))

    if "fields" in results and "text" in results:
        gap = results["fields"]["recall"] - results["text"]["recall"]
        print(
            f"\n两种模式的召回差 {gap * 100:+.1f}pp —— "
            "差值就是**文本抽取层**的损失，定位问题的价值就在这里。"
        )

    print(
        "\n注：合成集的 ground truth 由构造决定（缺陷是人埋的），"
        "所以这些数字是「引擎的相对能力」，不是「真实标签的绝对准确率」。"
        "这一点必须和 P1 的真实评测集分开表述。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
