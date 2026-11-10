"""给 P3 算指标：在合成卷宗上评一致性引擎。

    python scripts/eval_dossier.py --n 180

## 指标为什么这样设计

冲突检测的 **精确率** 比召回率更值钱：审核员被假冲突打扰三次之后就不看报告了。
所以样本里必须同时埋"真冲突"和"仅写法不同"两类，否则精确率毫无意义。

另外单独报**缺项召回**：缺章节是确定性规则，理应 100%。
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

from mc_dossier.consistency import review_dossier  # noqa: E402
from mc_dossier.schemas import ATTRIBUTE_LABELS  # noqa: E402
from mc_dossier.synthetic import DEFECT_LABELS, SyntheticDossier, generate  # noqa: E402


def evaluate(samples: list[SyntheticDossier]) -> dict:
    tp = fp = fn = 0
    clean_total = clean_fp = 0
    missing_total = missing_hit = 0
    per_kind_total: Counter[str] = Counter()
    per_kind_hit: Counter[str] = Counter()
    false_positive_attrs: Counter[str] = Counter()

    for sample in samples:
        result = review_dossier(sample.docs, dossier_id=sample.dossier_id)
        detected = {f.attribute for f in result.findings}
        planted = set(sample.planted_conflicts)

        hit = detected & planted
        extra = detected - planted
        tp += len(hit)
        fn += len(planted - hit)
        fp += len(extra)
        for attribute in extra:
            false_positive_attrs[ATTRIBUTE_LABELS[attribute]] += 1

        per_kind_total[sample.kind] += 1
        if not (planted - hit) and not extra:
            per_kind_hit[sample.kind] += 1

        if not planted and not sample.planted_missing:
            clean_total += 1
            if detected:
                clean_fp += 1

        if sample.planted_missing:
            missing_total += 1
            if set(sample.planted_missing) & set(result.missing_docs):
                missing_hit += 1

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
        "missing_total": missing_total,
        "missing_hit": missing_hit,
        "missing_recall": missing_hit / missing_total if missing_total else 0.0,
        "per_kind": {k: (per_kind_hit[k], per_kind_total[k]) for k in sorted(per_kind_total)},
        "false_positive_attrs": dict(false_positive_attrs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P3 卷宗一致性引擎评测")
    parser.add_argument("--n", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()

    samples = generate(args.n, seed=args.seed)
    m = evaluate(samples)

    print(f"合成卷宗 {len(samples)} 份（{len(DEFECT_LABELS)} 种情形，含负控）")
    print("=" * 62)
    print(f"冲突检测  P {m['precision']:.1%}   R {m['recall']:.1%}   F1 {m['f1']:.3f}")
    print(f"          TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(
        f"负控假阳性 {m['clean_fp']}/{m['clean_total']} = {m['clean_fp_rate']:.1%}"
        "  ← 只改写法不该报冲突"
    )
    print(f"缺项召回   {m['missing_hit']}/{m['missing_total']} = {m['missing_recall']:.1%}")

    if m["false_positive_attrs"]:
        print("\n  误报集中在：")
        for label, count in m["false_positive_attrs"].items():
            print(f"    {label}: {count}")

    print("\n  分情形（检出且无误报的比例）：")
    for kind, (hit, total) in m["per_kind"].items():
        flag = "" if hit == total else "  ← 未达 100%"
        print(f"    {DEFECT_LABELS.get(kind, kind):<26} {hit}/{total}{flag}")

    print(
        "\n注：合成集的 ground truth 由构造决定（冲突是人埋的），"
        "所以这些数字衡量的是「归一化是否既吃掉写法差异、又保留实质差异」，"
        "不是真实卷宗的绝对准确率。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
