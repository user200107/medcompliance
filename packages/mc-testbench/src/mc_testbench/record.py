"""运行记录：一次评测的完整快照，落盘为 JSON。

## 为什么记录里必须带数据指纹和 commit

两次 run 只有在这两项都相同时才**可比**。
不带指纹的对比是苹果比橘子，而人最容易忘记的恰恰是"我中间换过数据"。
所以指纹不是元数据装饰，它是比较的**前置条件**。
"""

from __future__ import annotations

import json
import secrets
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: 用东八区时间生成 run_id，跟报告里显示的时间一致，避免"报告写 15:32、id 是 07:32"的困惑
CST = timezone(timedelta(hours=8))

DEFAULT_RUNS_DIR = Path("runs")


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(CST)
    return f"{now:%Y%m%dT%H%M}-{secrets.token_hex(2)}"


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def git_commit(cwd: Path | None = None) -> str | None:
    """拿不到 git 信息不算错误（比如在导出的 zip 里跑），返回 None 即可。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


@dataclass
class RunRecord:
    run_id: str
    task: str
    created_at: str
    metrics: dict[str, float]
    config: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        missing = [k for k in ("run_id", "task", "metrics") if k not in data]
        if missing:
            raise ValueError(f"运行记录缺少字段：{', '.join(missing)}")
        return cls(
            run_id=data["run_id"],
            task=data["task"],
            created_at=data.get("created_at", ""),
            metrics={k: float(v) for k, v in data["metrics"].items()},
            config=data.get("config") or {},
            datasets=data.get("datasets") or {},
            meta=data.get("meta") or {},
        )


def save_run(record: RunRecord, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> Path:
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.run_id}.json"
    path.write_text(record.to_json(), encoding="utf-8")

    # 索引是追加写的：一次评测一行，便于 grep / 后续做趋势图
    index = directory / "index.jsonl"
    line = json.dumps(
        {
            "run_id": record.run_id,
            "task": record.task,
            "created_at": record.created_at,
            "config": record.config,
            "datasets": record.datasets,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with index.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def load_run(ref: str, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> RunRecord:
    """`ref` 可以是 run_id，也可以是 json 文件路径。"""
    candidate = Path(ref)
    if candidate.suffix == ".json" and candidate.is_file():
        return RunRecord.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
    path = Path(runs_dir) / f"{ref}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到运行记录：{ref}（在 {runs_dir} 下）")
    return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_runs(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> list[RunRecord]:
    directory = Path(runs_dir)
    if not directory.is_dir():
        return []
    records = [
        RunRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(directory.glob("*.json"))
    ]
    records.sort(key=lambda r: r.created_at)
    return records
