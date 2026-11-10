"""三个服务的端到端冒烟测试（需要先启动对应服务）。

    python scripts/smoke_all.py --base medireg=http://127.0.0.1:8011 \
                               --base labelguard=http://127.0.0.1:8012 \
                               --base dossier=http://127.0.0.1:8013
    python scripts/smoke_all.py --all-local      # 直接本地起三个（依赖 uvicorn）

不依赖 pytest，用于演示前快速确认"三个服务都活着且关键路径通"。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

LABEL = """产品名称：医用外科口罩
型号规格：17.5cm×9.5cm
注册人名称：某医疗器械有限公司
注册人住所：江苏省苏州市某路1号
联系方式：0512-12345678
医疗器械注册证编号：国械注准20243010123
生产日期：2026-01-01
失效日期：2029-01-01
贮存条件：常温干燥处保存
UDI：(01)06931234567890
无菌
一次性使用
注意事项：用后销毁
使用说明书
"""

DOSSIER = {
    "dossier_id": "smoke-d1",
    "docs": [
        {
            "doc_id": "t1",
            "kind": "technical_requirement",
            "filename": "产品技术要求.docx",
            "fields": {
                "model_spec": "A-100",
                "dimension": "Ø3.20 mm",
                "sterilization": "环氧乙烷灭菌",
                "shelf_life": "3年",
                "structure": "由针筒、推杆、活塞、针头组成",
                "manufacturer_address": "江苏省苏州市工业园区某路1号",
            },
        },
        {
            "doc_id": "t2",
            "kind": "ifu",
            "filename": "说明书.docx",
            "fields": {
                "model_spec": "A-100",
                "shelf_life": "36个月",
                "sterilization": "环氧乙烷",
                "manufacturer_address": "江苏省苏州市工业园区某路1号",
            },
        },
        {
            "doc_id": "t3",
            "kind": "test_report",
            "filename": "检验报告.pdf",
            "fields": {
                "model_spec": "A-100",
                "dimension": "Ø3.50 mm",  # ← 埋一个冲突
                "sterilization": "EO",
            },
        },
    ],
}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def check_medireg(base: str) -> None:
    h = _get(f"{base}/healthz")
    print(f"[P1] healthz ok，目录 {h['catalog_nodes']} 节点，抽取器 {h['normalizer']}")
    r = _post(
        f"{base}/api/v1/classify",
        {"description": "用于缓解肌肉疼痛的超声治疗设备，接触皮肤表面，需电源驱动", "risk": "balanced"},
    )
    best = r.get("best") or {}
    print(
        f"[P1] classify → status={r['status']} outcome={r.get('outcome')} "
        f"best={best.get('code', '-')} 风险偏好={r.get('risk_preference')}"
    )
    assert r["status"] == "classified", "P1 应当能给出结论"


def check_labelguard(base: str) -> None:
    h = _get(f"{base}/healthz")
    print(f"[P2] healthz ok，规则 {h['checklist_rules']} 条")

    r = _post(f"{base}/api/v1/labels/review", {"text": LABEL})
    print(
        f"[P2] review → status={r['status']} is_draft={r['is_draft']} "
        f"risk={r['review']['risk_level']} 违规={r['review']['violations']}"
    )
    assert r["is_draft"] is True

    draft_id = r["draft_id"]
    c = _post(f"{base}/api/v1/labels/drafts/{draft_id}/confirm", {"reviewer": "张审核", "revision": 1})
    print(f"[P2] confirm → status={c['status']} revision={c['revision']} 不可变={c['snapshot']['immutable']}")
    assert c["snapshot"]["immutable"] is True

    try:
        _post(f"{base}/api/v1/labels/drafts/{draft_id}/confirm", {"reviewer": "李四", "revision": 1})
    except urllib.error.HTTPError as exc:
        print(f"[P2] 重复确认被拒 → HTTP {exc.code}（revision 冲突）")
        assert exc.code == 409
    else:
        raise AssertionError("重复确认应当返回 409")


def check_dossier(base: str) -> None:
    h = _get(f"{base}/healthz")
    print(f"[P3] healthz ok，核对属性 {len(h['cross_checked_attributes'])} 项")
    r = _post(f"{base}/api/v1/dossiers/review", DOSSIER)
    print(f"[P3] review → 冲突 {r['conflicts']} 缺项 {len(r['missing_docs'])}")
    for f in r["findings"]:
        values = " vs ".join(f"「{o['raw_value']}」" for o in f["observations"])
        print(f"[P3]   [{f['severity']}] {f['label']}：{values}")
    assert r["conflicts"] == 1, "应当检出 1 处尺寸冲突"


def main() -> int:
    parser = argparse.ArgumentParser(description="三服务冒烟测试")
    parser.add_argument(
        "--base",
        action="append",
        default=[],
        help="name=url，可重复；name ∈ medireg/labelguard/dossier",
    )
    args = parser.parse_args()

    targets = {}
    for item in args.base:
        name, _, url = item.partition("=")
        targets[name] = url.rstrip("/")
    if not targets:
        print("至少给一个 --base，例如 --base labelguard=http://127.0.0.1:8012", file=sys.stderr)
        return 2

    checks = {"medireg": check_medireg, "labelguard": check_labelguard, "dossier": check_dossier}
    failures = 0
    for name, url in targets.items():
        if name not in checks:
            print(f"未知服务 {name}", file=sys.stderr)
            failures += 1
            continue
        try:
            checks[name](url)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1

    print("\n全部通过" if failures == 0 else f"\n{failures} 个服务未通过")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
