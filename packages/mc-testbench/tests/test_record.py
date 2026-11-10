"""运行记录的落盘与读取。"""

from __future__ import annotations

import json

import pytest

from mc_testbench.record import (
    RunRecord,
    list_runs,
    load_run,
    new_run_id,
    save_run,
)


def make(run_id="r1", task="toy", metrics=None) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        task=task,
        created_at="2026-11-10T17:00:00+08:00",
        metrics=metrics or {"recall": 0.8},
        config={"mode": "fields"},
        datasets={"toy_set": "v1"},
        meta={"git_commit": "abc1234"},
    )


def test_run_id_shape():
    run_id = new_run_id()
    assert len(run_id) == 18 and run_id[8] == "T"


def test_roundtrip(tmp_path):
    record = make()
    path = save_run(record, tmp_path)
    assert path.is_file()
    loaded = load_run("r1", tmp_path)
    assert loaded.metrics == record.metrics
    assert loaded.config == record.config
    assert loaded.datasets == record.datasets


def test_index_appended(tmp_path):
    save_run(make("r1"), tmp_path)
    save_run(make("r2"), tmp_path)
    lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "r1"


def test_list_runs_sorted(tmp_path):
    first = make("a")
    first.created_at = "2026-11-10T10:00:00+08:00"
    second = make("b")
    second.created_at = "2026-11-10T12:00:00+08:00"
    save_run(second, tmp_path)
    save_run(first, tmp_path)
    assert [r.run_id for r in list_runs(tmp_path)] == ["a", "b"]


def test_missing_field_rejected():
    with pytest.raises(ValueError, match="缺少字段"):
        RunRecord.from_dict({"run_id": "x"})


def test_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run("nope", tmp_path)
