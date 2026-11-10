"""命令行入口。

    python scripts/testbench.py run labels --mode fields
    python scripts/testbench.py list
    python scripts/testbench.py compare <base> <head>

`compare` 的退出码：0 无退化 / 1 有退化 / 2 不可比。
把它挂进 CI 就能挡住"指标悄悄变差"的合并。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mc_testbench.compare import EXIT_OK, compare, render
from mc_testbench.record import (
    DEFAULT_RUNS_DIR,
    RunRecord,
    git_commit,
    list_runs,
    load_run,
    new_run_id,
    now_iso,
    save_run,
)
from mc_testbench.tasks import TASKS, run


def _parse_config(pairs: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"参数需写成 key=value：{item}")
        key, _, raw = item.partition("=")
        config[key.strip()] = _coerce(raw.strip())
    return config


def _coerce(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _collect_config(args: argparse.Namespace) -> dict[str, Any]:
    """同时接受 `-c mode=fields` 与 `--mode fields`。

    文档和 README 里写的是 `--mode fields`，那更自然。与其改文档迁就实现，
    不如让实现接受它——**文档是承诺，代码该迁就文档**。
    """
    config = _parse_config(args.config)
    extra = list(getattr(args, "extra", []))
    index = 0
    while index < len(extra):
        token = extra[index]
        if not token.startswith("--"):
            raise SystemExit(f"无法识别的参数：{token}（配置请写成 key=value 或 --key value）")
        body = token[2:]
        if "=" in body:
            key, _, raw = body.partition("=")
            config[key] = _coerce(raw)
            index += 1
        else:
            if index + 1 >= len(extra):
                raise SystemExit(f"参数 {token} 缺少值")
            config[body] = _coerce(extra[index + 1])
            index += 2
    return config


def _cmd_run(args: argparse.Namespace, runs_dir: Path) -> int:
    outcome = run(args.task, **_collect_config(args))
    record = RunRecord(
        run_id=new_run_id(),
        task=args.task,
        created_at=now_iso(),
        metrics=outcome.metrics,
        config=outcome.config,
        datasets=outcome.datasets,
        meta={"git_commit": git_commit(), "python": sys.version.split()[0]},
    )
    path = save_run(record, runs_dir)
    print(f"[testbench] 已记录 {args.task} → {record.run_id}")
    print(f"[testbench] 记录文件：{path}")
    for key, value in record.metrics.items():
        spec = TASKS[args.task].metric(key)
        unit = spec.unit if spec else "count"
        label = spec.label if spec else key
        shown = f"{value:.1%}" if unit == "ratio" else f"{value:g}"
        print(f"  {label:<10} {shown}")
    return EXIT_OK


def _cmd_list(args: argparse.Namespace, runs_dir: Path) -> int:
    records = list_runs(runs_dir)
    if not records:
        print(f"[testbench] {runs_dir} 下没有运行记录")
        return EXIT_OK
    print(f"{'run_id':<22} {'task':<9} {'created_at':<20} config")
    print("-" * 78)
    for record in records:
        config = ", ".join(f"{k}={v}" for k, v in sorted(record.config.items()))
        print(f"{record.run_id:<22} {record.task:<9} {record.created_at:<20} {config}")
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace, runs_dir: Path) -> int:
    base = load_run(args.base, runs_dir)
    head = load_run(args.head, runs_dir)
    spec = TASKS.get(head.task)
    if spec is None:
        print(f"[testbench] 未知任务 {head.task}", file=sys.stderr)
        return 2
    comparison = compare(base, head, spec, tolerance=args.tolerance)
    print(render(comparison))
    return comparison.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testbench", description="评测运行管理与回归检测")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="跑一个任务并记录")
    p_run.add_argument("task", choices=sorted(TASKS))
    p_run.add_argument("--config", "-c", action="append", default=[], help="key=value，可重复")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="列出运行记录")
    p_list.set_defaults(func=_cmd_list)

    p_cmp = sub.add_parser("compare", help="对比两次运行，检测退化")
    p_cmp.add_argument("base")
    p_cmp.add_argument("head")
    p_cmp.add_argument("--tolerance", type=float, default=0.001, help="绝对值容差，默认 0.001")
    p_cmp.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # 用 parse_known_args：`--mode fields` 这类写法会被收集到 extra，由 _collect_config 处理
    args, extra = parser.parse_known_args(argv)
    args.extra = extra
    runs_dir = Path(args.runs_dir)
    return int(args.func(args, runs_dir))


if __name__ == "__main__":
    raise SystemExit(main())
