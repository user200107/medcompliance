"""抓取「医疗器械分类界定结果汇总」（官方逐条裁决）。

    python scripts/crawl_adjudications.py --pages 1-120
    python scripts/crawl_adjudications.py --all --delay 1.5

数据来源：中国食品药品检定研究院  分类界定结果汇总
    https://app.nidc.org.cn/biaogzx/dataGk.do?formAction=listFljdResult

**用途与边界（务必遵守）**
- 站点页脚声明「未经许可禁止转载或建立镜像」。因此本脚本抓到的数据
  **只能本地用于评测，不提交版本库、不对外发布、不再分发**。
  仓库里只保留本脚本、记录数/哈希/抓取日期，以及聚合指标。
- 输出目录 `data/raw/` 已在 .gitignore 中。
- 默认请求间隔 1.2s，带退避重试，支持断点续抓。不要把这个值调小。

字段映射：产品名称 + 产品详情 → 输入；产品类别 → 标签。
"""

from __future__ import annotations

import argparse
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

BASE = "https://app.nidc.org.cn/biaogzx/dataGk.do?formAction=listFljdResult&index={page}"
PAGE_SIZE = 10
UA = "Mozilla/5.0 (compatible; MedCompliance-portfolio-eval/0.1; local research use)"

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "adjudications.jsonl"


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
    """从分页脚本里取总页数：new Pagination(6434, 10, 1, ...) → 644。"""
    m = re.search(r"new Pagination\((\d+),\s*(\d+),\s*(\d+)", html)
    if not m:
        return None
    total, per_page = int(m.group(1)), int(m.group(2))
    return -(-total // per_page)


def parse_rows(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="tab1")
    if table is None:
        return []
    body = table.find("tbody") or table

    out: list[dict[str, str]] = []
    for tr in body.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 6 or cells[0] == "序号":
            continue
        # 序号 | 产品名称 | 产品类别 | 产品详情 | 标题 | 发布日期 | 备注
        out.append(
            {
                "seq": cells[0],
                "product_name": cells[1],
                "product_category": cells[2],
                "detail": cells[3],
                "notice_title": cells[4],
                "publish_date": cells[5],
            }
        )
    return out


def record_id(row: dict[str, str]) -> str:
    key = f"{row['notice_title']}|{row['product_name']}|{row['detail']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def parse_page_range(spec: str) -> list[int]:
    """'1-120' / '5' / '1-3,7'"""
    pages: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(chunk))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取分类界定结果（仅供本地评测）")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--pages", default="1-120", help="如 1-120 或 1-3,7（默认 1-120）")
    parser.add_argument("--all", action="store_true", help="抓全部页（以页面公布的总数为准）")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="从第 1 页到末页每 N 页取 1 页，用于跨全时间段的等距抽样（优先于 --pages）",
    )
    parser.add_argument("--delay", type=float, default=1.2, help="请求间隔秒，别调小")
    args = parser.parse_args()

    if args.delay < 0.5:
        print("警告：--delay 低于 0.5s 对公开服务不友好，已强制为 0.5", file=sys.stderr)
        args.delay = 0.5

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.out)
    print(f"已有记录 {len(done)} 条，写入 {args.out}")

    # 先取第 1 页，确定总页数并验证解析
    first_html = fetch_page(1)
    total_pages = parse_total_pages(first_html)
    print(f"服务端报告总页数：{total_pages}")
    if not parse_rows(first_html):
        raise SystemExit("解析第 1 页得到 0 行 —— HTML 结构可能变了，先别继续")

    pages = list(range(1, (total_pages or 1) + 1)) if args.all else parse_page_range(args.pages)
    if args.sample_every > 1:
        # 站内是按发布时间正序的：只抓前 N 页会得到一个「全是某一年」的偏样本。
        # 等距抽样才能覆盖全部年份与全部公告批次。
        last = total_pages or max(pages)
        pages = list(range(1, last + 1, args.sample_every))
    print(f"本次计划抓取 {len(pages)} 页（{len(pages) * PAGE_SIZE} 条上限）")

    fetched = parse_rows(first_html) if 1 in pages else []
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
                row["source_url"] = BASE.format(page=page)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        if fetched:
            emit(fetched, 1)

        todo = [p for p in pages if p != 1]
        for i, page in enumerate(todo, 1):
            time.sleep(args.delay)
            try:
                html = fetch_page(page)
            except RuntimeError as exc:
                print(f"  ! 跳过第 {page} 页：{exc}", file=sys.stderr)
                continue
            rows = parse_rows(html)
            if not rows:
                print(f"  ! 第 {page} 页解析为 0 行，跳过", file=sys.stderr)
                continue
            emit(rows, page)
            if i % 20 == 0:
                fh.flush()
                print(f"  进度 {i}/{len(todo)} 页，累计新增 {written} 条")

    print(f"完成：新增 {written} 条，文件共 {len(done)} 条 → {args.out}")


if __name__ == "__main__":
    main()
