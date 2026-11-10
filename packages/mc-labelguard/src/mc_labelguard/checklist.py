"""必标项规则引擎。

## 设计主张

1. **规则是数据。** 加一个审查项 = 加一行 `RULE()`，判定引擎不动。
   这也是"法规条款驱动产品"的具体形态。
2. **判定分三态。** 缺项/格式错/与主数据不符 → `violation`；
   看不清/没抽到 → `undeterminable`。**低置信度绝不允许被判成违规或合规。**
3. **每条判定带条款。** 报告里要能指出"凭什么"。
4. **条件是模型的一部分。** 无菌标识只对无菌产品要求，一次性使用标识只对一次性产品要求；
   条件不满足时该项为 `compliant`（不适用），并在理由里写明。

## 依据

- 《医疗器械说明书和标签管理规定》（原总局令第 6 号）
- YY/T 0466.1（医疗器械 用于医疗器械标签、标记和提供信息的符号）
- 《医疗器械唯一标识系统规则》

> 条款号在真实产品里应向 `mc_domain.models.Clause` 查；本模块先用常量字符串
> 占位，待法规库导入后改为外键引用。这个替换点是刻意留出来的。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from mc_domain.enums import RequirementKey

from mc_labelguard.schemas import (
    ChecklistItemResult,
    ItemStatus,
    LabelExtraction,
    LabelReviewResult,
    RiskLevel,
)

CLAUSE_BASE = "《医疗器械说明书和标签管理规定》"
DEFAULT_MIN_CONFIDENCE = 0.6

#: 注册证编号的结构校验。**只做浅层结构检查，不做注册库查证**——
#: 真伪必须连国家局数据库，不能假装能查。这一点在报告里会写明。
_CERT_NO_RE = re.compile(r"^(国械注准|国械注进|国械备|[一-龥]械注准|[一-龥]械备)\d{9,11}$")
_UDI_RE = re.compile(r"^\(01\)\d{14}")
_DATE_ORDER_KIND = "date_order"


@dataclass(frozen=True)
class Rule:
    key: RequirementKey
    label: str
    clause: str
    kind: Literal["presence", "pattern", "date_order"]
    required: bool = True
    #: 返回 True 表示该项适用；不适用时判 compliant 并注明
    applies_when: Callable[[LabelExtraction], bool] | None = None
    pattern: re.Pattern[str] | None = None
    pattern_hint: str = ""
    #: 条件项的"为什么要求"说明，用在不适用的理由里
    condition_hint: str = ""


def _sterile(e: LabelExtraction) -> bool:
    """条件看的是**产品属性**，不是符号有没有标。"""
    return e.is_sterile is True


def _single_use(e: LabelExtraction) -> bool:
    return e.is_single_use is True


RULES: tuple[Rule, ...] = (
    Rule(RequirementKey.LABEL_PRODUCT_NAME, "产品名称", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(RequirementKey.LABEL_MODEL_SPEC, "型号规格", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(RequirementKey.LABEL_REGISTRANT_NAME, "注册人/备案人名称", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(RequirementKey.LABEL_REGISTRANT_ADDRESS, "注册人/备案人住所", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(RequirementKey.LABEL_REGISTRANT_CONTACT, "联系方式", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(
        RequirementKey.LABEL_CERT_NO,
        "医疗器械注册证编号",
        f"{CLAUSE_BASE} 第十条",
        "pattern",
        pattern=_CERT_NO_RE,
        pattern_hint="形如 国械注准20243010123 / 苏械注准20242020001",
    ),
    Rule(
        RequirementKey.LABEL_PRODUCTION_LICENSE,
        "生产许可证/备案编号",
        f"{CLAUSE_BASE} 第十条 及配套生产许可要求",
        "presence",
        required=False,
    ),
    Rule(RequirementKey.LABEL_MANUFACTURE_DATE, "生产日期", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(RequirementKey.LABEL_EXPIRY_DATE, "失效日期/使用期限", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(RequirementKey.LABEL_STORAGE, "贮存条件", f"{CLAUSE_BASE} 第十条", "presence"),
    Rule(
        RequirementKey.LABEL_UDI,
        "唯一标识（UDI）",
        "《医疗器械唯一标识系统规则》",
        "pattern",
        pattern=_UDI_RE,
        pattern_hint="GS1 格式形如 (01)0693xxxxxxxxxx",
    ),
    Rule(
        RequirementKey.LABEL_STERILE_MARK,
        "无菌标识（YY/T 0466.1 符号）",
        "YY/T 0466.1",
        "presence",
        applies_when=_sterile,
        condition_hint="产品为无菌提供时才要求",
    ),
    Rule(
        RequirementKey.LABEL_SINGLE_USE,
        "一次性使用标识（YY/T 0466.1 符号）",
        "YY/T 0466.1",
        "presence",
        applies_when=_single_use,
        condition_hint="产品为一次性使用时才要求",
    ),
    Rule(
        RequirementKey.LABEL_WARNING,
        "警示信息",
        f"{CLAUSE_BASE} 第十条",
        "presence",
        required=False,
    ),
    Rule(
        RequirementKey.LABEL_IFU_REF,
        "说明书指引",
        f"{CLAUSE_BASE} 第十条",
        "presence",
        required=False,
    ),
)


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _may_judge(extraction: LabelExtraction, min_confidence: float) -> tuple[bool, str]:
    """能不能对「缺失字段」下违规结论。

    不能判有两种原因：置信度不足、图像质量有问题。两种都必须落到 undeterminable。
    """
    if extraction.confidence < min_confidence:
        return False, f"抽取置信度 {extraction.confidence:.2f} 低于阈值 {min_confidence:.2f}"
    if extraction.quality_flags:
        return False, "图像质量提示：" + "、".join(extraction.quality_flags)
    return True, ""


def check_label(
    extraction: LabelExtraction,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    as_of: str | None = None,
) -> LabelReviewResult:
    can_judge, block_reason = _may_judge(extraction, min_confidence)
    items: list[ChecklistItemResult] = []

    for rule in RULES:
        if rule.applies_when is not None and not rule.applies_when(extraction):
            items.append(
                ChecklistItemResult(
                    key=rule.key,
                    label=rule.label,
                    required=False,
                    status=ItemStatus.COMPLIANT,
                    reason=f"本项目不适用（{rule.condition_hint}）",
                    clause=rule.clause,
                )
            )
            continue

        value = extraction.get(rule.key)
        if _is_blank(value):
            if not rule.required:
                # 非强制项缺失**不是违规**。判成违规会制造假阳性，
                # 让企业去整改一个法规并不要求它标注的项。
                items.append(
                    ChecklistItemResult(
                        key=rule.key,
                        label=rule.label,
                        required=False,
                        status=ItemStatus.COMPLIANT,
                        reason="非强制项，未标注不构成违规",
                        clause=rule.clause,
                    )
                )
                continue

            status = ItemStatus.VIOLATION if can_judge else ItemStatus.UNDETERMINABLE
            if rule.key.value in extraction.parse_issues:
                # 标注了、但解析不出合法值：这是**格式问题**，不是缺项。
                # 混为一谈会给企业错误的整改方向（"去补标注" vs "去改格式"）。
                reason = (
                    f"已标注但未能识别为合法格式，应满足 {rule.pattern_hint}"
                    if can_judge
                    else f"疑似已标注但未能识别：{block_reason}"
                )
            else:
                reason = "标签未标注该项" if can_judge else f"未能确认该项：{block_reason}"
            items.append(
                ChecklistItemResult(
                    key=rule.key,
                    label=rule.label,
                    required=rule.required,
                    status=status,
                    reason=reason,
                    clause=rule.clause,
                )
            )
            continue

        if rule.kind == "pattern" and isinstance(value, str):
            ok = bool(rule.pattern and rule.pattern.match(value.strip()))
            if not ok:
                items.append(
                    ChecklistItemResult(
                        key=rule.key,
                        label=rule.label,
                        required=rule.required,
                        status=ItemStatus.VIOLATION,
                        reason=f"格式不符：{value.strip()}；应满足 {rule.pattern_hint}",
                        clause=rule.clause,
                    )
                )
                continue

        items.append(
            ChecklistItemResult(
                key=rule.key,
                label=rule.label,
                required=rule.required,
                status=ItemStatus.COMPLIANT,
                reason="已标注",
                clause=rule.clause,
            )
        )

    # 日期先后：生产日期不得晚于失效日期
    if extraction.manufacture_date and extraction.expiry_date:
        if extraction.manufacture_date > extraction.expiry_date:
            items.append(
                ChecklistItemResult(
                    key=RequirementKey.LABEL_EXPIRY_DATE,
                    label="生产日期与失效日期先后",
                    required=True,
                    status=ItemStatus.VIOLATION,
                    reason=(
                        f"生产日期 {extraction.manufacture_date} 晚于失效日期 "
                        f"{extraction.expiry_date}"
                    ),
                    clause=f"{CLAUSE_BASE} 第十条",
                )
            )

    counts = {s: sum(1 for i in items if i.status is s) for s in ItemStatus}

    required_violation = any(
        i.status is ItemStatus.VIOLATION and i.required for i in items
    )
    any_violation = counts[ItemStatus.VIOLATION] > 0
    high_undeterminable = (counts[ItemStatus.UNDETERMINABLE] / len(items)) > 0.4 if items else False

    if required_violation:
        risk = RiskLevel.HIGH
    elif any_violation or high_undeterminable:
        risk = RiskLevel.MEDIUM
    elif counts[ItemStatus.UNDETERMINABLE] > 0:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.LOW

    caveats = [
        "本结论由 AI 自动核对生成，须经审核员复核确认后方可对外出具。",
        "注册证编号仅做结构校验，未与国家药监局数据库比对，不构成真伪证明。",
    ]
    if not can_judge:
        caveats.append(
            f"因{block_reason}，缺失项一律判为「不可判定」，未按违规处理。请补拍清晰照片后重新核对。"
        )
    if high_undeterminable:
        caveats.append(
            f"不可判定项占比 {counts[ItemStatus.UNDETERMINABLE] / len(items):.0%}，"
            "结论可靠性不足，建议人工逐项核对。"
        )

    return LabelReviewResult(
        risk_level=risk,
        items=items,
        compliant=counts[ItemStatus.COMPLIANT],
        violations=counts[ItemStatus.VIOLATION],
        undeterminable=counts[ItemStatus.UNDETERMINABLE],
        caveats=caveats,
    )


__all__ = ["DEFAULT_MIN_CONFIDENCE", "RULES", "Rule", "check_label"]
