"""属性值的归一化——P3 的技术核心。

## 为什么这是核心而不是细节

原始实现最容易犯的错是**直接字符串比对**，于是：

    技术要求：Ø3.20 mm      检验报告：3.2mm         → 误报冲突
    技术要求：型号 A-100     说明书：A100           → 误报冲突
    技术要求：有效期 3 年    说明书：36 个月         → 误报冲突

而真正该报的冲突是：

    技术要求：Ø3.20 mm      检验报告：Ø3.50 mm      → 必须报

**误报比漏报更能摧毁信任**：审核员连续三次被假冲突打扰，就会不再看报告。
所以归一化的目标是「把写法差异吃掉，把实质差异留下」，而不是"尽量宽松"。

## 归一化策略（每条都能举反例，所以都写清边界）

- 型号：去空格/连字符/全角，统一大写。**不做模糊匹配**——A100 与 A-100 视为同一，
  但 A100 与 A101 必须是不同（那是两个型号，正是要抓的）。
- 尺寸：解析「数值 + 单位」，统一换算到毫米。**保留原始精度**，
  3.20 与 3.2 相等（数值比较），3.20 与 3.5 不等。
- 有效期：解析「N 年 / N 个月 / 具体日期」，统一换算成月数。
- 灭菌方式：同义词归并（EO / 环氧乙烷 / 环氧乙烷灭菌 → 同一）。
- 地址、结构组成：只做空白与标点归一，**不做同义词归并**——
  地址"江苏省苏州市"与"苏州"确实可能不同，不能替企业裁定。
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from mc_dossier.schemas import Attribute

_WS = re.compile(r"[\s\u3000]+")
_FULLWIDTH = str.maketrans({"（": "(", "）": ")", "：": ":", "，": ",", "／": "/"})

_DIMENSION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|um|μm|毫米|厘米)")
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*年")
_MONTHS_RE = re.compile(r"(\d+)\s*个?月")
_ISO_DATE = re.compile(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})")

_STERILIZATION_CANON = (
    ("环氧乙烷", "环氧乙烷"),
    ("eo", "环氧乙烷"),
    ("eto", "环氧乙烷"),
    ("辐照", "辐照"),
    ("radiation", "辐照"),
    ("gamma", "辐照"),
    ("钴60", "辐照"),
    ("湿热", "湿热灭菌"),
    ("蒸汽", "湿热灭菌"),
    ("高压灭菌", "湿热灭菌"),
    ("干热", "干热灭菌"),
    ("灭菌", "未注明方式"),
)


def _basic(text: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", text.translate(_FULLWIDTH))).lower()


def normalize_model(text: str) -> str:
    """型号：去分隔符 + 大写。A-100 / A 100 / ａ100 → a100。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fa5]", "", _basic(text))


def normalize_dimension(text: str) -> str | None:
    """尺寸统一到毫米，保留三位小数以吃掉 3.2 与 3.20 的写法差异。

    **没有单位时返回 None（不可比对）**，而不是退回原文。
    理由：`3.2` 到底是 3.2mm 还是 3.2cm，工具无权假定。
    退回原文会把它和 `Ø3.20 mm` 判成"冲突"——那是在制造假冲突。
    """
    flat = unicodedata.normalize("NFKC", text).lower()
    m = _DIMENSION_RE.search(flat)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit in ("cm", "厘米"):
        value *= 10
    elif unit in ("um", "μm"):
        value /= 1000
    return f"{value:.3f}mm"


def normalize_shelf_life(text: str) -> str | None:
    """统一换算成月数；具体日期用 `date:` 前缀标记，不与月数互比。

    解析不出"N 年 / N 月 / 具体日期"时返回 None（不可比对）——
    例如只写"长期"，那不是一个可比较的期限。
    """
    flat = unicodedata.normalize("NFKC", text)

    if (m := _ISO_DATE.search(flat)):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"date:{d.isoformat()}"
        except ValueError:
            pass

    if (m := _YEARS_RE.search(flat)):
        return f"months:{int(round(float(m.group(1)) * 12))}"
    if (m := _MONTHS_RE.search(flat)):
        return f"months:{int(m.group(1))}"
    return None


def normalize_sterilization(text: str) -> str:
    flat = _basic(text)
    for needle, canon in _STERILIZATION_CANON:
        if needle in flat:
            return canon
    return flat


def normalize_text_value(text: str) -> str:
    """地址 / 结构组成：只吃掉空白与标点差异，**不做同义词归并**。

    替企业裁定"苏州"与"江苏省苏州市"是同一地址，超出工具的能力边界。
    """
    return re.sub(r"[^0-9a-z\u4e00-\u9fa5]", "", _basic(text))


_NORMALIZERS = {
    Attribute.MODEL_SPEC: normalize_model,
    Attribute.DIMENSION: normalize_dimension,
    Attribute.SHELF_LIFE: normalize_shelf_life,
    Attribute.STERILIZATION: normalize_sterilization,
    Attribute.MANUFACTURER_ADDRESS: normalize_text_value,
    Attribute.STRUCTURE: normalize_text_value,
}


def normalize(attribute: Attribute, value: str) -> str:
    return _NORMALIZERS[attribute](value)


__all__ = [
    "normalize",
    "normalize_dimension",
    "normalize_model",
    "normalize_shelf_life",
    "normalize_sterilization",
    "normalize_text_value",
]
