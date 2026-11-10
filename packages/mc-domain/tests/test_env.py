"""`.env` 加载器的单测。

这条链路最容易出的问题是**静默失效**：文件写对了但没加载，
程序用默认值跑了一路，最后"为什么没调模型"查半天。所以要有测试。
"""

from __future__ import annotations

import os

import pytest

from mc_domain.env import find_dotenv, load_dotenv, parse_dotenv


def test_parse_basic():
    text = """
# 注释
MC_LLM_MODEL=qwen-plus

MC_LLM_API_KEY=sk-abc123
EMPTY=
"""
    parsed = parse_dotenv(text)
    assert parsed == {"MC_LLM_MODEL": "qwen-plus", "MC_LLM_API_KEY": "sk-abc123", "EMPTY": ""}


def test_parse_strips_quotes():
    assert parse_dotenv('A="x y"\nB=\'z\'\n')["A"] == "x y"
    assert parse_dotenv("B='z'")["B"] == "z"


def test_parse_ignores_malformed_lines():
    parsed = parse_dotenv("没有等号的一行\n=没有键\nOK=1\n")
    assert parsed == {"OK": "1"}


def test_parse_keeps_equals_in_value():
    assert parse_dotenv("URL=https://x/y?a=1&b=2")["URL"] == "https://x/y?a=1&b=2"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MC_TEST_KEY=from_file\n", encoding="utf-8")

    monkeypatch.setenv("MC_TEST_KEY", "from_shell")
    load_dotenv(env_file)
    assert os.environ["MC_TEST_KEY"] == "from_shell"

    monkeypatch.delenv("MC_TEST_KEY")
    load_dotenv(env_file)
    assert os.environ["MC_TEST_KEY"] == "from_file"
    monkeypatch.delenv("MC_TEST_KEY")


def test_load_dotenv_missing_file_returns_none(tmp_path):
    assert load_dotenv(tmp_path / "不存在.env") is None


def test_find_dotenv_walks_up(tmp_path):
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    found = find_dotenv(nested)
    assert found is not None and found.parent == tmp_path


@pytest.mark.parametrize("key", ["MC_LLM_MODEL", "MC_RERANK_MODEL", "MC_EMBEDDING_MODEL"])
def test_repo_env_example_parses(key):
    """仓库里的 .env.example 必须能被自己的解析器读出来，否则文档就是错的。"""
    from pathlib import Path

    example = Path(__file__).resolve().parents[3] / ".env.example"
    assert example.is_file(), f"缺少 {example}"
    parsed = parse_dotenv(example.read_text(encoding="utf-8"))
    assert key in parsed, f".env.example 里缺少 {key}"
