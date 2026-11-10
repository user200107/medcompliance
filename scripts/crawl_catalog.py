"""抓取「分类目录库」——真实的《医疗器械分类目录》结构化数据。

    python scripts/crawl_catalog.py --all --delay 1.0
    python scripts/crawl_catalog.py --all --delay 1.0 --csv data/raw/catalog_real.csv

数据来源：中国食品药品检定研究院  分类目录库
    https://app.nidc.org.cn/biaogzx/dataGk.do?formAction=listFlmlk
共 1,703 条，10 条/页，171 页。

字段：子目录 | 一级产品类别 | 二级产品类别 | 产品描述 | 预期用途 | 品名举例 | 管理类别

## 为什么不直接 F 载官方公告的附件

官方原始目录在《总局关于发布医疗器械分类目录的公告》（2017 年第 104 号）的附件里，
是 Word/PDF，需要自己转表、自己处理跨页断行与合并单元格。而这个库是同一份内容的
结构化视图，直接可解析。**要留官方存档可以另外下，但不必用它来喂程序。**

## 编码规则（重要）

官方编号是三段式，但**每段的编号都从 1 重新开始**：
    子目录 `01` → 一级 `01`（在该子目录内）→ 二级 `01.1`（在该一级内）
所以同一段编号在不同父节点下会重复。为了得到全局唯一的树，
这里拼成 `01` / `01-01` / `01-01-01.1`，`name` 里保留官方原文。

## 用途与边界

- 站点声明「未经许可禁止转载或建立镜像」：数据只本地使用，不提交仓库、不再分发。
- 默认请求间隔 1.0s，带退避重试与断点续抓。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://app.nidc.org.cn/biaogzx/dataGk.do?formAction=listFlmlk&index={page}"
PAGE_SIZE = 10
UA = "Mozilla/5.0 (compatible; MedCompliance-portfolio-eval/0.1; local research use)"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "catalog_real.jsonl"

#: 官方编号形如 `01 有源手术器械` / `01.1 超声手术设备`
_CODE_NAME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(.+?)\s*$")
_WS_RE = re.compile(r"[\s\u3000\xa0]+")


def fetch_page(page: int, *, timeout: int = 30, retries: int = 3) -> str:
    url = BASE.format(page=page)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return raw.decode("gb18030", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            wait = 2.0 * (attempt + 1) + random.uniform(0, 0.5)
            print(f"  ! page {page} 第 {attempt + 1} 次失败：{exc}，{wait:.1f}s 后重试", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"page {page} 抓取失败：{last}")


def parse_total_pages(html: str) -> int | None:
    m = re.search(r"new Pagination\((\d+),\s*(\d+),\s*(\d+)", html)
    if not m:
        return None
    total, per_page = int(m.group(1)), int(m.group(2))
    return -(-total // per_page)


def _split_code_name(cell: str) -> tuple[str, str]:
    text = _WS_RE.sub(" ", cell).strip()
    m = _CODE_NAME_RE.match(text)
    if m:
        return m.group(1), m.group(2)
    return "", text


def parse_rows(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="tab1")
    if table is None:
        return []
    body = table.find("tbody") or table

    out: list[dict[str, str]] = []
    for tr in body.find_all("tr"):
        cells = [td.get_text(strip=False) for td in tr.find_all("td")]
        # 序号 | 子目录 | 一级产品类别 | 二级产品类别 | 产品描述 | 预期用途 | 品名举例 | 管理类别
        if len(cells) < 8 or _WS_RE.sub("", cells[0]) == "序号":
            continue

        sub_code, sub_name = _split_code_name(cells[1])
        l1_code, l1_name = _split_code_name(cells[2])
        l2_code, l2_name = _split_code_name(cells[3])

        out.append(
            {
                "sub_code": sub_code,
                "sub_name": sub_name,
                "l1_code": l1_code,
                "l1_name": l1_name,
                "l2_code": l2_code,
                "l2_name": l2_name,
                "description": _WS_RE.sub(" ", cells[4]).strip(),
                "intended_use": _WS_RE.sub(" ", cells[5]).strip(),
                "examples": _WS_RE.sub(" ", cells[6]).strip(),
                "management_class": _WS_RE.sub("", cells[7]).strip(),
            }
        )
    return out


def record_id(row: dict[str, str]) -> str:
    key = "|".join(
        [row["sub_code"], row["l1_code"], row["l2_code"], row["l2_name"], row["description"]]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------------------------------------------------------------------------
# 转成 catalog_loader 认的 CSV
# ---------------------------------------------------------------------------

_CLASS_MAP = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "I": "I", "II": "II", "III": "III"}


def to_catalog_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """官方三段编号 → 全局唯一的树节点。

    官方每段编号都从 1 重开（一级在自己的子目录内编号、二级在自己的一级内编号），
    所以拼成 `01` / `01-01` / `01-01-01.1` 才能保证全局唯一。
    官方原编号保留在 `name` 里，不影响可读性。
    """
    nodes: dict[str, dict[str, object]] = {}

    for row in rows:
        cls = _CLASS_MAP.get(row["management_class"])
        if cls is None:
            continue

        sub = row["sub_code"]
        l1 = f"{sub}-{row['l1_code']}" if row["l1_code"] else None
        l2 = f"{l1}-{row['l2_code']}" if (l1 and row["l2_code"]) else None

        if sub and sub not in nodes:
            nodes[sub] = {
                "code": sub,
                "name": row["sub_name"],
                "parent_code": "",
                "management_class": "",
                "definition": "",
                "scope_note": "",
                "exclusion_note": "",
                "examples": "",
                "sort_order": _sort_key(sub),
            }
        if l1 and l1 not in nodes:
            nodes[l1] = {
                "code": l1,
                "name": row["l1_name"],
                "parent_code": sub,
                "management_class": "",
                "definition": "",
                "scope_note": "",
                "exclusion_note": "",
                "examples": "",
                "sort_order": _sort_key(l1),
            }
        if l2:
            # 同一二级编码可能被多条记录共享（不同品名举例），合并举例与描述
            existing = nodes.get(l2)
            examples = [e for e in row["examples"].replace("，", "、").split("、") if e.strip()]
            if existing:
                merged = set(str(existing["examples"]).split("|")) | set(examples)
                existing["examples"] = "|".join(sorted(x for x in merged if x))
            else:
                nodes[l2] = {
                    "code": l2,
                    "name": row["l2_name"],
                    "parent_code": l1,
                    "management_class": cls,
                    "definition": row["description"],
                    "scope_note": row["intended_use"],
                    "exclusion_note": "",
                    "examples": "|".join(examples),
                    "sort_order": _sort_key(l2),
                }

    return sorted(nodes.values(), key=lambda r: (len(str(r["code"]).split("-")), _sort_key(str(r["code"]))))


def _sort_key(code: str) -> int:
    """把 `01-02-03.1` 编成一个可排序的整数。

    `catalog_loader.load_csv` 会把 sort_order 读成 int，
    所以这里必须返回数字而不是字符串——否则转出来的 CSV 会在导入时炸。
    每段按两位加权：01-02-03.1 → 1*10^6 + 2*10^4 + 3*10^2 + 1 = 1020301
    """
    parts = [p for p in re.split(r"[-.]", code) if p.isdigit()]
    weights = (1_000_000, 10_000, 100, 1)
    total = 0
    for part, weight in zip(parts[:4], weights):
        total += min(int(part), 99) * weight
    return total


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "code", "name", "parent_code", "management_class", "definition",
        "scope_note", "exclusion_note", "examples", "sort_order",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取分类目录库（仅供本地使用）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=None, help="同时输出 catalog_loader 可读的 CSV")
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--pages", default="")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="不抓取，只把已有 JSONL 转成 CSV（调格式时用，避免重抓）",
    )
    args = parser.parse_args()

    if args.convert_only:
        all_rows = [json.loads(line) for line in args.out.open(encoding="utf-8") if line.strip()]
        nodes = to_catalog_rows(all_rows)
        target = args.csv or (args.out.parent / "catalog_real.csv")
        write_csv(nodes, target)
        print(f"已写出目录 CSV：{target}（{len(nodes)} 个节点，来自 {len(all_rows)} 条记录）")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.out)
    print(f"已有 {len(done)} 条 → {args.out}")

    first_html = fetch_page(1)
    total_pages = parse_total_pages(first_html) or 1
    print(f"总页数：{total_pages}")

    pages = list(range(1, total_pages + 1))
    if args.pages:
        pages = list(range(1, 1))

    fetched = parse_rows(first_html)
    written = 0
    with args.out.open("a", encoding="utf-8") as fh:

        def emit(rows: list[dict[str, str]], page: int) -> None:
            nonlocal written
            for row in rows:
                rid = record_id(row)
                if rid in done:
                    continue
                done.add(rid)
                row["id"] = rid
                row["page"] = page
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        emit(fetched, 1)
        for i, page in enumerate(pages[1:], 1):
            time.sleep(args.delay)
            try:
                html = fetch_page(page)
            except RuntimeError as exc:
                print(f"  ! 跳过第 {page} 页：{exc}", file=sys.stderr)
                continue
            rows = parse_rows(html)
            if not rows:
                print(f"  ! 第 {page} 页解析为 0 行", file=sys.stderr)
                continue
            emit(rows, page)
            if i % 20 == 0:
                fh.flush()
                print(f"  进度 {i}/{len(pages) - 1}，累计 {written} 条")

    print(f"完成：新增 {written} 条，文件共 {len(done)} 条")

    if args.csv:
        all_rows = [json.loads(line) for line in args.out.open(encoding="utf-8") if line.strip()]
        nodes = to_catalog_rows(all_rows)
        write_csv(nodes, args.csv)
        print(f"已写出目录 CSV：{args.csv}（{len(nodes)} 个节点）")


if __name__ == "__main__":
    main()
