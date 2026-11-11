"""Unit tests for src.env (.env loading)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.env import load_env


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("VT_TEST_A", "VT_TEST_B", "VT_TEST_JA"):
        monkeypatch.delenv(name, raising=False)


def test_missing_file_returns_false(tmp_path: Path) -> None:
    assert load_env(tmp_path / ".env") is False


def test_loads_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nVT_TEST_A=hello\nVT_TEST_JA=日本語\n", encoding="utf-8"
    )
    assert load_env(env_file) is True
    assert os.environ["VT_TEST_A"] == "hello"
    assert os.environ["VT_TEST_JA"] == "日本語"
    # load_dotenv writes to os.environ directly; register for cleanup.
    monkeypatch.delenv("VT_TEST_A")
    monkeypatch.delenv("VT_TEST_JA")


def test_existing_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VT_TEST_B", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("VT_TEST_B=from-file\n", encoding="utf-8")
    load_env(env_file)
    assert os.environ["VT_TEST_B"] == "from-shell"
