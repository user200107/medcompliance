"""把自由文本规范化为 `ProductProfile`。

这是整条流水线里**唯一允许调用模型**的阶段。设计上有两条铁律：

1. **失败必须抛异常，绝不返回空画像。**
   上一代项目（FourfoldClarity）踩过这个坑：视觉模型解析失败返回 `{}`，
   随后 `setdefault("confidence", 0.5)` 把它包装成"成功但为空"的结果，
   直接流进了下游判定。等价于把"模型挂了"变成了"专业结论"。
   这里用 `NormalizationFailed` 显式断开这条路。

2. **模型只做翻译，不做判定。**
   它输出的是"这个产品有没有接触人体、有没有源"，不是"它是几类"。
   定类永远由 `mc_domain.classify` 的确定性阶段完成。

两个实现：
- `HeuristicNormalizer` 纯关键词，离线可跑，是降级路径与单测基线；
- `LLMNormalizer` 走 OpenAI 兼容协议，解析失败/超时/字段非法一律抛错。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

import httpx
from mc_domain.env import http_client_kwargs
from mc_domain.enums import ContactDuration, ContactType, EnergyType
from mc_domain.schemas import ProductProfile

logger = logging.getLogger(__name__)


class NormalizationFailed(RuntimeError):
    """槽位抽取失败。调用方应返回 5xx 并提示重试，**不得**降级成空画像继续判定。"""


class ProfileNormalizer(Protocol):
    async def normalize(self, raw_text: str) -> ProductProfile: ...


# ---------------------------------------------------------------------------
# 关键词基线
# ---------------------------------------------------------------------------

#: 顺序敏感：先判"无源/被动"，否则「不依赖电源」里的「电源」会误判为有源。
_PASSIVE_HINTS = ("无源", "被动", "不依赖电源", "不含电源", "手工操作", "手动")
_ACTIVE_HINTS = ("有源", "电源", "电池", "供电", "电动", "驱动", "通电", "插电", "电机", "电磁")

_IMPLANT_HINTS = ("植入", "埋入体内", "植入体内")
_INVASIVE_HINTS = ("侵入", "穿刺", "插入体内", "进入人体内部", "进入血管", "经血管", "手术器械")
_INDIRECT_HINTS = ("间接接触", "管路", "输液", "血液回路", "经体液", "与体液")
_SURFACE_HINTS = ("表面接触", "皮肤", "黏膜", "体表", "接触人体表面", "表面")
_NONE_CONTACT_HINTS = ("不接触人体", "体外", "不接触患者", "不接触人体表面")

_LONG_HINTS = ("长期", "30天以上", "三十天以上", "永久植入", "持续植入")
_SHORT_HINTS = ("数小时", "几天", "数日", "短期", "一周内")
_TRANSIENT_HINTS = ("短时间内", "几分钟", "数分钟", "暂时")

_STERILE_TRUE = ("无菌", "灭菌")
_STERILE_FALSE = ("非无菌", "不灭菌")
_REUSABLE_TRUE = ("可重复使用", "复用", "可重复灭菌")
_REUSABLE_FALSE = ("一次性使用", "一次性", "不得重复使用")

_SENTENCE_END = re.compile(r"[。；;\n]")
_PURPOSE_MARKERS = ("用于", "适用于", "供")


def _first_hit(text: str, hints: tuple[str, ...]) -> str | None:
    for h in hints:
        if h in text:
            return h
    return None


def _extract_purpose(text: str) -> str | None:
    """取「用于……」那一段作为预期用途。取不到就退回全文。"""
    for marker in _PURPOSE_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            tail = text[idx:]
            m = _SENTENCE_END.search(tail)
            segment = tail[: m.start()] if m else tail
            if len(segment) >= len(marker) + 2:
                return segment.strip()
    return text.strip() or None


class HeuristicNormalizer:
    """关键词抽取。**这是降级路径，不是生产主路径。**

    能力边界写在明处：只认表里列出的词，同义改写、否定、组合描述都会漏。
    漏掉的槽位保持 `None`，由流水线转成追问——这是正确行为，不是 bug。
    它能覆盖的是"用户照着说明书念"这类规整输入。
    """

    async def normalize(self, raw_text: str) -> ProductProfile:
        text = raw_text.strip()
        if not text:
            raise NormalizationFailed("产品描述为空，无法抽取槽位")

        energy: EnergyType | None = None
        if _first_hit(text, _PASSIVE_HINTS):
            energy = EnergyType.PASSIVE
        elif _first_hit(text, _ACTIVE_HINTS):
            energy = EnergyType.ACTIVE

        contact: ContactType | None = None
        if _first_hit(text, _NONE_CONTACT_HINTS):
            contact = ContactType.NONE
        elif _first_hit(text, _IMPLANT_HINTS):
            contact = ContactType.IMPLANT
        elif _first_hit(text, _INVASIVE_HINTS):
            contact = ContactType.INVASIVE
        elif _first_hit(text, _INDIRECT_HINTS):
            contact = ContactType.INDIRECT
        elif _first_hit(text, _SURFACE_HINTS):
            contact = ContactType.SURFACE

        duration: ContactDuration | None = None
        if contact is None or contact is ContactType.NONE:
            duration = ContactDuration.NONE
        elif _first_hit(text, _LONG_HINTS):
            duration = ContactDuration.LONG
        elif _first_hit(text, _SHORT_HINTS):
            duration = ContactDuration.SHORT
        elif _first_hit(text, _TRANSIENT_HINTS):
            duration = ContactDuration.TRANSIENT

        sterile: bool | None = None
        if _first_hit(text, _STERILE_FALSE):
            sterile = False
        elif _first_hit(text, _STERILE_TRUE):
            sterile = True

        reusable: bool | None = None
        if _first_hit(text, _REUSABLE_FALSE):
            reusable = False
        elif _first_hit(text, _REUSABLE_TRUE):
            reusable = True

        return ProductProfile(
            name=None,
            intended_use=_extract_purpose(text),
            energy_type=energy,
            contact_type=contact,
            contact_duration=duration,
            is_sterile=sterile,
            is_reusable=reusable,
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# LLM 实现
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是医疗器械注册技术文档助手。请从用户描述中抽取结构化字段，只输出 JSON。

字段与取值（不在下列取值内的一律返回 null，不要猜）：
- intended_use: 字符串，产品的预期用途原文
- name: 字符串，产品名称
- contact_type: "none" | "surface" | "indirect" | "invasive" | "implant"
  none=不接触人体；surface=表面接触；indirect=经管路或体液间接接触；
  invasive=经体表进入体内；implant=植入体内
- contact_duration: "none" | "transient" | "short" | "long"
  transient<60分钟；short<30天；long>=30天
- energy_type: "active" | "passive"（是否依赖电源/产生能量）
- is_sterile: true | false | null
- is_reusable: true | false | null
- materials: 字符串数组
- models: 字符串数组

硬性要求：
1. 用户没说的一律返回 null，**禁止推断或补全**。
2. 只输出 JSON，不要解释、不要 Markdown 代码块。
3. 不要输出管理类别、分类编码或任何监管结论——那不属于你的职责。"""

_ALLOWED_TOP_KEYS = {
    "intended_use", "name", "contact_type", "contact_duration",
    "energy_type", "is_sterile", "is_reusable", "materials", "models",
}


def _extract_json(text: str) -> dict[str, Any]:
    """从可能带代码块/前后缀的回复里取出 JSON 对象。取不到就抛。"""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise NormalizationFailed("模型回复中找不到 JSON 对象") from None
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise NormalizationFailed(f"模型回复不是合法 JSON：{exc}") from exc
    if not isinstance(obj, dict):
        raise NormalizationFailed(f"模型回复应为 JSON 对象，实际为 {type(obj).__name__}")
    return obj


def _coerce_lists(data: dict[str, Any]) -> dict[str, Any]:
    """把列表字段里的 null 归一成 []。

    提示词明确要求"用户没说的一律返回 null"，所以模型对 `models` / `materials`
    返回 null 是**预期行为**，不是错误。标量字段的 null 表示"未抽取"，
    列表字段的 null 则表示"没有条目"——两者语义不同，不能都当校验失败。
    之前这里直接 `model_validate`，导致真实调用第一条就抛 NormalizationFailed。
    """
    out = dict(data)
    for key in ("materials", "models"):
        if out.get(key) is None:
            out[key] = []
        elif isinstance(out.get(key), str):
            out[key] = [out[key]] if out[key].strip() else []
    return out


class LLMNormalizer:
    """OpenAI 兼容协议的槽位抽取。

    任何一步失败（网络、HTTP、JSON、字段校验）都抛 `NormalizationFailed`，
    **不重试成空画像**。重试只针对可恢复错误，且次数有上限。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: int = 30,
        max_retries: int = 2,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_s
        self._max_retries = max_retries
        #: 累计 token 消耗，便于控制评测成本。只在成功的响应上累加。
        self.total_tokens = 0
        self.calls = 0
        self.failures = 0

    async def normalize(self, raw_text: str) -> ProductProfile:
        if not raw_text.strip():
            raise NormalizationFailed("产品描述为空，无法抽取槽位")

        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_text.strip()},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        last_error: Exception | None = None
        async with httpx.AsyncClient(**http_client_kwargs(self._timeout)) as client:  # type: ignore[arg-type]
            for attempt in range(self._max_retries + 1):
                try:
                    resp = await client.post(self._url, json=payload, headers=headers)
                    resp.raise_for_status()
                    body = resp.json()
                    self.calls += 1
                    usage = body.get("usage") or {}
                    self.total_tokens += int(usage.get("total_tokens") or 0)
                    content = body["choices"][0]["message"]["content"]
                    data = _extract_json(content)
                    return self._to_profile(data, raw_text)
                except NormalizationFailed:
                    self.failures += 1
                    raise
                except Exception as exc:  # noqa: BLE001 — 统一转成领域异常
                    last_error = exc
                    logger.warning("槽位抽取第 %d 次失败：%s", attempt + 1, exc)
                    if attempt < self._max_retries:
                        await asyncio.sleep(0.4 * (attempt + 1))

        self.failures += 1
        raise NormalizationFailed(f"槽位抽取失败（已重试 {self._max_retries} 次）：{last_error}")

    def _to_profile(self, data: dict[str, Any], raw_text: str) -> ProductProfile:
        unknown = set(data) - _ALLOWED_TOP_KEYS
        if unknown:
            logger.info("模型返回了未定义字段，已忽略：%s", sorted(unknown))
        cleaned = {k: v for k, v in data.items() if k in _ALLOWED_TOP_KEYS}
        cleaned = _coerce_lists(cleaned)
        cleaned["raw_text"] = raw_text

        try:
            profile = ProductProfile.model_validate(cleaned)
        except Exception as exc:  # noqa: BLE001
            raise NormalizationFailed(f"模型输出未通过字段校验：{exc}") from exc

        # 一个字段都没抽出来，说明模型实质失败，不要伪装成"用户没说"
        if not (profile.intended_use or profile.name):
            raise NormalizationFailed("模型未抽取到预期用途或产品名称，视为抽取失败")
        return profile


def build_normalizer(
    kind: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    timeout_s: int = 30,
    max_retries: int = 2,
) -> ProfileNormalizer:
    if kind == "llm":
        return LLMNormalizer(base_url, api_key, model, timeout_s=timeout_s, max_retries=max_retries)
    return HeuristicNormalizer()
