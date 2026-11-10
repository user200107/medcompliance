"""P3 HTTP 层与"不可比对"语义的单测。

两条容易退化的约定：
1. 尺寸/有效期归一化不出可比值时必须判「无法核对」，**不能退回原文比字符串**
   （那会把 `3.2` 和 `Ø3.20 mm` 判成冲突 = 假阳性）；
2. 接口在只给 `text` 时应能自己抽字段（auto_extract）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mc_dossier.consistency import review_dossier
from mc_dossier.main import create_app
from mc_dossier.schemas import Attribute, DocKind, Document
from mc_dossier.synthetic import build_dossier


def _client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def client():
    with _client() as c:
        yield c


# ---------------------------------------------------------------------------
# 不可比对
# ---------------------------------------------------------------------------


def test_unit_less_dimension_is_unverifiable_not_conflict():
    """`3.2` 没有单位，工具无权假定是 mm —— 应判无法核对，不能判冲突。"""
    dossier = build_dossier("hard_rewording", dossier_id="h1")
    result = review_dossier(dossier.docs)
    assert result.conflicts == 0
    assert Attribute.DIMENSION in result.unverifiable
    assert Attribute.DIMENSION not in result.consistent_attributes


def test_shelf_life_without_number_is_unverifiable():
    docs = [
        Document(doc_id="a", kind=DocKind.TECHNICAL_REQUIREMENT, filename="a", fields={"shelf_life": "3年"}),
        Document(doc_id="b", kind=DocKind.IFU, filename="b", fields={"shelf_life": "长期"}),
    ]
    result = review_dossier(docs)
    assert result.conflicts == 0
    assert Attribute.SHELF_LIFE in result.unverifiable


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["implemented"] is True
    assert body["stateless"] is True
    assert "technical_requirement" in body["required_docs"]


def test_requirements_endpoint_lists_attributes(client):
    body = client.get("/api/v1/dossiers/requirements").json()
    assert len(body["attributes"]) == 6
    assert all(a["label"] for a in body["attributes"])


def test_review_detects_conflict(client):
    dossier = build_dossier("dimension_mismatch", dossier_id="d1")
    payload = {
        "dossier_id": "d1",
        "docs": [
            {
                "doc_id": d.doc_id,
                "kind": d.kind.value,
                "filename": d.filename,
                "fields": d.fields,
            }
            for d in dossier.docs
        ],
    }
    body = client.post("/api/v1/dossiers/review", json=payload).json()
    assert body["conflicts"] == 1
    finding = body["findings"][0]
    assert finding["attribute"] == "dimension"
    assert finding["severity"] == "high"
    # 报告必须同时给出两边的原值
    assert len(finding["observations"]) == 2
    assert body["review_required"] is True


def test_review_auto_extract_from_text(client):
    """只给 text 时接口应自己抽字段——这是接真实卷宗解析的入口。"""
    payload = {
        "docs": [
            {
                "doc_id": "t1",
                "kind": "technical_requirement",
                "text": "型号规格：A-100\n关键尺寸：Ø3.20 mm\n灭菌方式：环氧乙烷灭菌\n有效期：3年\n",
            },
            {
                "doc_id": "t3",
                "kind": "test_report",
                "text": "型号规格：A-100\n关键尺寸：Ø3.50 mm\n灭菌方式：环氧乙烷\n",
            },
        ]
    }
    body = client.post("/api/v1/dossiers/review", json=payload).json()
    assert body["conflicts"] == 1
    assert body["findings"][0]["attribute"] == "dimension"


def test_missing_doc_reported(client):
    dossier = build_dossier("missing_test_report", dossier_id="d2")
    payload = {
        "docs": [
            {"doc_id": d.doc_id, "kind": d.kind.value, "filename": d.filename, "fields": d.fields}
            for d in dossier.docs
        ]
    }
    body = client.post("/api/v1/dossiers/review", json=payload).json()
    assert "test_report" in body["missing_docs"]


def test_empty_docs_rejected(client):
    assert client.post("/api/v1/dossiers/review", json={"docs": []}).status_code == 422
