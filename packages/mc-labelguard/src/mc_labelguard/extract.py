"""从文本抽取标签字段。

**为什么先做文本模式而不是直接上视觉模型**

1. 标签审核的核心难点是**判定**（三态 + 条款依据），不是 OCR。判定引擎可以被
   完整单测覆盖，视觉抽取只能靠人工看效果；
2. 文本模式让我可以在**合成标签**上构造带已知缺陷的评测集——缺陷是人为埋的，
   所以 ground truth 是确定的，能算出准确率；
3. 视觉抽取是**可替换的前端**：接 qwen-vl 时只要产出同样的 `LabelExtraction`，
   下游一行不用改。这条边界是刻意留出的。

`confidence` 的语义：文本模式视为"已可靠读取"，给 0.9——
即缺失字段应当判违规（而不是不可判定）。视觉模式才需要按图像质量下调。
"""

from __future__ import annotations

import re
from datetime import date

from mc_labelguard.schemas import LabelExtraction

_CERT_NO = re.compile(r"(国械注准|国械注进|国械备|[一-龥]械注准|[一-龥]械备)\s*(\d{9,11})")
_UDI = re.compile(r"\(01\)\s*(\d{14})")

_DATE = r"(\d{4})[-年./](\d{1,2})[-月./](\d{1,2})"
_MANUFACTURE = re.compile(r"生产日期[：:\s]*" + _DATE)
_EXPIRY = re.compile(r"(?:失效日期|有效期至|使用期限至)[：:\s]*" + _DATE)


def _field(text: str, *names: str) -> str | None:
    for name in names:
        m = re.search(rf"{name}[：:\s]*([^\n，,；;]+)", text)
        if m:
            value = m.group(1).strip(" 　：:。.")
            if value:
                return value
    return None


def _to_date(match: re.Match[str] | None) -> date | None:
    if not match:
        return None
    y, mo, d = (int(g) for g in match.groups()[:3])
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def extract_from_text(text: str) -> LabelExtraction:
    raw = text or ""
    flat = raw.replace("\u3000", " ")

    cert = _CERT_NO.search(flat)
    udi = _UDI.search(flat)

    # 「非无菌」「不得重复使用」这类否定表述必须先判，否则会被正向关键词吃掉
    sterile: bool | None = None
    if "非无菌" in flat or "不灭菌" in flat:
        sterile = False
    elif "无菌" in flat:
        sterile = True

    single_use: bool | None = None
    if "可重复使用" in flat or "复用" in flat:
        single_use = False
    elif "一次性使用" in flat or "一次性" in flat:
        single_use = True

    # 文本模式下「无菌」既是产品属性也是符号存在，二者都置位；
    # 视觉模式可以只看到符号而不知道产品属性，那时二者会分开。

    issues: list[str] = []
    if cert is None and ("注册证编号" in flat or "注册证号" in flat):
        issues.append("label.cert_no")
    if udi is None and "UDI" in flat.upper():
        issues.append("label.udi")

    return LabelExtraction(
        product_name=_field(flat, "产品名称", "品名"),
        model_spec=_field(flat, "型号规格", "规格型号", "型号"),
        registrant_name=_field(flat, "注册人名称", "注册人", "备案人名称", "备案人"),
        registrant_address=_field(flat, "注册人住所", "住所", "注册地址", "生产地址"),
        registrant_contact=_field(flat, "联系方式", "联系电话", "电话"),
        cert_no=f"{cert.group(1)}{cert.group(2)}" if cert else None,
        production_license=_field(flat, "生产许可证编号", "生产许可证", "生产备案凭证编号"),
        manufacture_date=_to_date(_MANUFACTURE.search(flat)),
        expiry_date=_to_date(_EXPIRY.search(flat)),
        storage=_field(flat, "贮存条件", "储存条件", "贮藏条件"),
        udi=f"(01){udi.group(1)}" if udi else None,
        is_sterile=sterile,
        is_single_use=single_use,
        sterile_mark=sterile,
        single_use_mark=single_use,
        warning=_field(flat, "警示", "注意事项", "警示信息"),
        ifu_ref=("使用说明书" if ("使用说明书" in flat or "详见说明书" in flat) else None),
        confidence=0.9,
        raw_text=raw,
        parse_issues=issues,
    )


__all__ = ["extract_from_text"]
