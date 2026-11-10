"""合成标签集：人为埋入已知缺陷，用于给 P2 算出真实指标。

## 为什么合成集在这里是合理的

标签审核的判定标准是**可枚举的法定清单**，所以缺陷可以按项精确埋入：
"少标注册证编号"就是 `cert_no=None`，期望检出的项就是 `label.cert_no`。
ground truth 由构造决定，不存在标注分歧。

这与 P1 的情况不同——P1 的"管理类别"没人能凭构造保证，所以只能靠官方裁决。
**能用合成集的前提是"判定标准可枚举"，这一点本身就该讲清楚。**

## 设计

- 每条样本先构造一个合规的基线字段集（`_baseline`）；
- 再植入**恰好一个**缺陷（或零个，作为负控）；
- `planted` 记录期望检出的违规项，`clean=True` 表示期望零违规；
- 同时输出 text 形式，用于验证文本抽取链路。

固定 seed，保证可复现。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

from mc_domain.enums import RequirementKey

from mc_labelguard.schemas import LabelExtraction

DEFECT_RENDERERS: dict[str, str] = {
    "missing_cert_no": "缺注册证编号",
    "malformed_cert_no": "注册证号格式错",
    "missing_udi": "缺 UDI",
    "missing_storage": "缺贮存条件",
    "missing_product_name": "缺产品名称",
    "missing_registrant_address": "缺注册人住所",
    "date_order": "生产日期晚于失效日期",
    "sterile_missing_symbol": "无菌产品未标无菌符号",
    "single_use_missing_symbol": "一次性产品未标一次性使用符号",
    "clean": "无缺陷（负控）",
}

#: 每个缺陷埋入后，期望被检出的 RequirementKey
EXPECTED: dict[str, tuple[RequirementKey, ...]] = {
    "missing_cert_no": (RequirementKey.LABEL_CERT_NO,),
    "malformed_cert_no": (RequirementKey.LABEL_CERT_NO,),
    "missing_udi": (RequirementKey.LABEL_UDI,),
    "missing_storage": (RequirementKey.LABEL_STORAGE,),
    "missing_product_name": (RequirementKey.LABEL_PRODUCT_NAME,),
    "missing_registrant_address": (RequirementKey.LABEL_REGISTRANT_ADDRESS,),
    "date_order": (RequirementKey.LABEL_EXPIRY_DATE,),
    "sterile_missing_symbol": (RequirementKey.LABEL_STERILE_MARK,),
    "single_use_missing_symbol": (RequirementKey.LABEL_SINGLE_USE,),
    "clean": (),
}

_PRODUCTS = (
    ("一次性使用无菌注射器", "2ml", True, True),
    ("医用外科口罩", "17.5cm×9.5cm", True, True),
    ("电子血压计", "HEM-7136", False, False),
    ("医用退热贴", "50mm×120mm", False, True),
    ("一次性使用输液器", "带针 1.2mm", True, True),
    ("医用弹力绷带", "7.5cm×4.5m", False, True),
)


@dataclass
class SyntheticLabel:
    label_id: str
    fields: LabelExtraction
    planted: tuple[RequirementKey, ...]
    defect_kind: str
    text: str = field(default="")

    @property
    def clean(self) -> bool:
        return not self.planted


def _baseline(rng: random.Random) -> LabelExtraction:
    name, spec, is_sterile, is_single_use = rng.choice(_PRODUCTS)
    year = rng.choice((2024, 2025, 2026))
    return LabelExtraction(
        product_name=name,
        model_spec=spec,
        registrant_name="某医疗器械有限公司",
        registrant_address="江苏省苏州市某路 1 号",
        registrant_contact="0512-12345678",
        cert_no=f"国械注准{year}{rng.randint(1000000, 9999999)}",
        production_license="苏食药监械生产许20010001号",
        manufacture_date=date(2026, 1, 1),
        expiry_date=date(2029, 1, 1),
        storage="常温干燥处保存",
        udi="(01)06931234567890",
        is_sterile=is_sterile,
        is_single_use=is_single_use,
        sterile_mark=is_sterile,
        single_use_mark=is_single_use,
        warning="一次性使用，用后销毁",
        ifu_ref="使用说明书",
        confidence=0.9,
    )


def _apply(kind: str, e: LabelExtraction) -> LabelExtraction:
    if kind == "missing_cert_no":
        return e.model_copy(update={"cert_no": None})
    if kind == "malformed_cert_no":
        return e.model_copy(update={"cert_no": "国械注准2024"})
    if kind == "missing_udi":
        return e.model_copy(update={"udi": None})
    if kind == "missing_storage":
        return e.model_copy(update={"storage": None})
    if kind == "missing_product_name":
        return e.model_copy(update={"product_name": None})
    if kind == "missing_registrant_address":
        return e.model_copy(update={"registrant_address": None})
    if kind == "date_order":
        return e.model_copy(
            update={"manufacture_date": date(2030, 1, 1), "expiry_date": date(2029, 1, 1)}
        )
    if kind == "sterile_missing_symbol":
        # 产品是无菌的，但标签上没标符号 —— 这是拆分字段后才抓得到的缺陷
        return e.model_copy(update={"is_sterile": True, "sterile_mark": None})
    if kind == "single_use_missing_symbol":
        return e.model_copy(update={"is_single_use": True, "single_use_mark": None})
    return e


def to_text(e: LabelExtraction) -> str:
    """把字段渲染成标签文本，用于验证抽取链路（缺失字段自然不出现）。"""
    lines: list[str] = []
    if e.product_name:
        lines.append(f"产品名称：{e.product_name}")
    if e.model_spec:
        lines.append(f"型号规格：{e.model_spec}")
    if e.registrant_name:
        lines.append(f"注册人名称：{e.registrant_name}")
    if e.registrant_address:
        lines.append(f"注册人住所：{e.registrant_address}")
    if e.registrant_contact:
        lines.append(f"联系方式：{e.registrant_contact}")
    if e.cert_no:
        lines.append(f"医疗器械注册证编号：{e.cert_no}")
    if e.production_license:
        lines.append(f"生产许可证编号：{e.production_license}")
    if e.manufacture_date:
        lines.append(f"生产日期：{e.manufacture_date}")
    if e.expiry_date:
        lines.append(f"失效日期：{e.expiry_date}")
    if e.storage:
        lines.append(f"贮存条件：{e.storage}")
    if e.udi:
        lines.append(f"UDI：{e.udi}")
    if e.sterile_mark:
        lines.append("无菌")
    if e.single_use_mark:
        lines.append("一次性使用")
    if e.warning:
        lines.append(f"注意事项：{e.warning}")
    if e.ifu_ref:
        lines.append("使用说明书")
    return "\n".join(lines) + "\n"


def generate(n: int, *, seed: int = 20261110) -> list[SyntheticLabel]:
    """生成 n 条样本，缺陷类型均匀轮转；约 1/6 为负控（无缺陷）。"""
    rng = random.Random(seed)
    kinds = [k for k in DEFECT_RENDERERS if k != "clean"]
    out: list[SyntheticLabel] = []
    for i in range(n):
        kind = "clean" if i % 6 == 5 else kinds[i % len(kinds)]
        base = _baseline(rng)
        fields = _apply(kind, base)
        out.append(
            SyntheticLabel(
                label_id=f"syn-{i:04d}",
                fields=fields,
                planted=EXPECTED[kind],
                defect_kind=kind,
                text=to_text(fields),
            )
        )
    return out


__all__ = ["DEFECT_RENDERERS", "EXPECTED", "SyntheticLabel", "generate", "to_text"]
