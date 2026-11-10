"""P1 HTTP 层测试。

重点不是"路由能通"，而是三条语义：
1. 槽位够 → 给出结论 + 依据链；
2. 槽位不够 → 返回追问，**不是**报错，也不是瞎猜；
3. 抽取失败 → 5xx，**不是**伪装成成功的空结果。
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mc_medireg.config import Settings
from mc_medireg.main import create_app
from mc_medireg.normalizer import HeuristicNormalizer, LLMNormalizer, NormalizationFailed

#: 合成目录的绝对路径。测试固定用它，避免依赖开发者本地是否存在真实目录。
SAMPLE_CATALOG = Path(str(resources.files("mc_domain").joinpath("data/catalog_sample.csv")))


@pytest.fixture(scope="module")
def client():
    """显式固定为合成目录。

    `Settings()` 的默认目录会**优先找 `data/raw/catalog_real.csv`**（为了演示效果），
    所以测试绝不能依赖默认值——否则"某个开发者本地碰巧有真实目录"就会让断言失效。
    测试要把前提写死，这是稳定性问题不是风格问题。
    """
    app = create_app(
        Settings(
            catalog_csv=SAMPLE_CATALOG,
            catalog_version="sample-synthetic",
            normalizer="heuristic",
        )
    )
    with TestClient(app) as c:
        yield c


def test_healthz_reports_catalog_state(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["catalog_nodes"] == 8
    assert body["classifiable_nodes"] == 4  # 07-01-01 / 07-01-02 / 09-01 / 01-01
    assert body["normalizer"] == "heuristic"


def test_classify_full_description_gives_conclusion(client):
    resp = client.post(
        "/api/v1/classify",
        json={"description": "用于缓解肌肉疼痛的超声治疗设备，接触皮肤表面，需电源驱动"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "classified"
    assert body["best"]["code"] == "09-01"
    assert body["best"]["management_class"] == "III"
    # 依据链必须完整：先过闸门，排除过的候选也要能看见
    assert any(r["code"] == "07" for r in body["rejected"])
    assert [s["stage"] for s in body["steps"]][:3] == ["normalize", "gate", "coarse"]
    assert body["outcome"] == "III"


def test_sparse_description_concludes_but_flags_unconfirmed(client):
    """信息不全不再拦人：给结论 + 明确列出未确认维度 + 保留追问入口。"""
    resp = client.post("/api/v1/classify", json={"description": "眼科超声诊断设备"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "classified"
    assert body["best"] is not None
    # 未确认维度必须回传，且写进 caveats
    assert "contact_type" in body["unconfirmed"]
    assert any("未确认" in c for c in body["caveats"])
    # 追问入口仍在，只是不再阻断
    assert body["missing_slots"]
    assert len(body["missing_slots"]) <= 2


def test_classify_with_profile_skips_extraction(client):
    resp = client.post(
        "/api/v1/classify/profile",
        json={
            "profile": {
                "name": "眼科超声诊断设备",
                "intended_use": "用于眼科检查，观察眼底",
                "contact_type": "surface",
                "energy_type": "active",
            }
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["best"]["code"].startswith("07-01")


def test_catalog_node_found_and_missing(client):
    ok = client.get("/api/v1/catalog/09-01")
    assert ok.status_code == 200
    assert ok.json()["name"] == "超声治疗设备"
    assert ok.json()["name_path"] == "物理治疗器械/超声治疗设备"

    missing = client.get("/api/v1/catalog/99-99")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "catalog_node_not_found"


def test_as_of_is_echoed_back(client):
    resp = client.post(
        "/api/v1/classify",
        json={
            "description": "用于缓解肌肉疼痛的超声治疗设备，接触皮肤表面，需电源驱动",
            "as_of": "2024-06-01",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["as_of"] == "2024-06-01"


# ---------------------------------------------------------------------------
# 失败语义：这组测试保护的是「不许把失败伪装成成功」
# ---------------------------------------------------------------------------


def test_risk_presets_listed(client):
    body = client.get("/api/v1/risk-presets").json()
    assert body["default"] == "balanced"
    names = {p["name"] for p in body["presets"]}
    assert {"strict", "balanced", "risk_averse", "max_recall_iii"} <= names


def test_invalid_risk_rejected(client):
    resp = client.post(
        "/api/v1/classify",
        json={"description": "眼科超声诊断设备", "risk": "拍脑袋"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "invalid_risk_preference"


def test_risk_is_echoed_in_result(client):
    resp = client.post(
        "/api/v1/classify",
        json={"description": "眼科超声诊断设备", "risk": "risk_averse"},
    )
    assert resp.status_code == 200
    assert resp.json()["risk_preference"] == "risk_averse"


def test_empty_description_raises_not_returns_empty_profile():
    import asyncio

    with pytest.raises(NormalizationFailed):
        asyncio.run(HeuristicNormalizer().normalize("   "))


def test_llm_profile_with_no_fields_is_treated_as_failure():
    """模型返回合法 JSON 但全为 null —— 必须当失败处理，不能当成"用户没说"。"""
    normalizer = LLMNormalizer("http://example.invalid", "k", "m")
    with pytest.raises(NormalizationFailed):
        normalizer._to_profile({"intended_use": None, "name": None}, "某产品")


def test_llm_profile_rejects_unknown_keys_but_keeps_known():
    normalizer = LLMNormalizer("http://example.invalid", "k", "m")
    profile = normalizer._to_profile(
        {"intended_use": "用于诊断", "management_class": "III", "乱写": 1}, "原文"
    )
    assert profile.intended_use == "用于诊断"
    assert not hasattr(profile, "management_class")

def test_null_list_fields_are_coerced_not_rejected():
    """模型按提示词返回 models/materials 为 null 是预期行为，必须归一成 []。

    这条是真实调用第一颗子弹打出来的：之前直接 model_validate，
    结果第一条样本就抛 NormalizationFailed，整个评测跑不起来。
    """
    normalizer = LLMNormalizer("http://example.invalid", "k", "m")
    profile = normalizer._to_profile(
        {"intended_use": "用于诊断", "models": None, "materials": None}, "原文"
    )
    assert profile.models == []
    assert profile.materials == []
    assert profile.intended_use == "用于诊断"


def test_string_list_field_becomes_single_item_list():
    normalizer = LLMNormalizer("http://example.invalid", "k", "m")
    profile = normalizer._to_profile({"name": "X", "models": "A-100"}, "原文")
    assert profile.models == ["A-100"]
