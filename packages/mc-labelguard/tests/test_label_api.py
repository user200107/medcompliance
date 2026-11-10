"""P2 HTTP 层单测。

重点不是路由通不通，而是**发布门禁的三条硬约束**：
1. 送审只返回草稿（`is_draft: true`），不直接给"报告"；
2. 确认必须署名，否则等于 AI 直接出报告；
3. 基于旧 revision 的确认必须 409，不能覆盖别人的复核。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mc_labelguard.main import create_app

FULL_LABEL = """产品名称：一次性使用无菌注射器
型号规格：2ml
注册人名称：某医疗器械有限公司
注册人住所：江苏省苏州市某路 1 号
联系方式：0512-12345678
医疗器械注册证编号：国械注准20243010123
生产许可证编号：苏食药监械生产许20010001号
生产日期：2026-01-01
失效日期：2029-01-01
贮存条件：常温干燥处保存
UDI：(01)06931234567890
无菌
一次性使用
注意事项：用后销毁
使用说明书
"""


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_healthz_reports_rules(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["checklist_rules"] >= 15
    assert body["implemented"] is True


def test_checklist_exposed_with_clauses(client):
    body = client.get("/api/v1/labels/checklist").json()
    assert body["count"] >= 15
    assert all(item["clause"] for item in body["items"])
    keys = {i["key"] for i in body["items"]}
    assert "label.cert_no" in keys


def test_review_returns_draft_not_report(client):
    body = client.post("/api/v1/labels/review", json={"text": FULL_LABEL}).json()
    assert body["is_draft"] is True
    assert body["status"] == "pending_review"
    assert body["revision"] == 1
    assert body["review"]["review_required"] is True
    assert body["review"]["violations"] == 0


def test_missing_item_produces_violation(client):
    text = FULL_LABEL.replace("医疗器械注册证编号：国械注准20243010123\n", "")
    body = client.post("/api/v1/labels/review", json={"text": text}).json()
    assert body["review"]["risk_level"] == "high"
    keys = [i["key"] for i in body["review"]["items"] if i["status"] == "violation"]
    assert "label.cert_no" in keys


def test_empty_input_rejected(client):
    resp = client.post("/api/v1/labels/review", json={"text": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "empty_input"


def test_confirm_requires_reviewer_name(client):
    draft = client.post("/api/v1/labels/review", json={"text": FULL_LABEL}).json()
    resp = client.post(
        f"/api/v1/labels/drafts/{draft['draft_id']}/confirm",
        json={"reviewer": "   ", "revision": 1},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "invalid_reviewer"


def test_confirm_produces_immutable_snapshot(client):
    draft = client.post("/api/v1/labels/review", json={"text": FULL_LABEL}).json()
    body = client.post(
        f"/api/v1/labels/drafts/{draft['draft_id']}/confirm",
        json={"reviewer": "张审核", "revision": 1, "note": "已复核"},
    ).json()

    assert body["status"] == "confirmed"
    assert body["revision"] == 2
    assert body["reviewer"] == "张审核"
    assert body["snapshot"]["immutable"] is True
    assert body["snapshot"]["reviewer"] == "张审核"


def test_stale_revision_conflicts(client):
    draft = client.post("/api/v1/labels/review", json={"text": FULL_LABEL}).json()
    url = f"/api/v1/labels/drafts/{draft['draft_id']}/confirm"
    client.post(url, json={"reviewer": "甲", "revision": 1})

    resp = client.post(url, json={"reviewer": "乙", "revision": 1})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "revision_conflict"
    assert resp.json()["detail"]["current_revision"] == 2


def test_unknown_draft_404(client):
    assert client.get("/api/v1/labels/drafts/nope").status_code == 404


def test_structured_fields_input_supported(client):
    """视觉抽取接上后走的就是这条路：直接给结构化字段，不经过文本正则。"""
    body = client.post(
        "/api/v1/labels/review",
        json={
            "fields": {
                "product_name": "某器械",
                "model_spec": "A-1",
                "registrant_name": "某公司",
                "registrant_address": "某地址",
                "registrant_contact": "000-0000",
                "cert_no": "国械注准20243010123",
                "manufacture_date": "2026-01-01",
                "expiry_date": "2029-01-01",
                "storage": "常温",
                "udi": "(01)06931234567890",
                "confidence": 0.9,
            }
        },
    ).json()
    # 缺少生产许可证（非必需）与警示/说明书（非必需），但都必须有条款依据
    assert body["review"]["review_required"] is True
    assert body["review"]["violations"] == 0
