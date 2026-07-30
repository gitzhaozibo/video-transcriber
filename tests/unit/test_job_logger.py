"""Unit tests for src.job_logger (JSON Lines logging)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(tmp_path: Path):
    """Return a fresh JobLogger writing to tmp_path."""
    import importlib
    import sys

    # Force re-import so _HANDLER_INSTALLED resets and we can set LOG_DIR.
    for mod_name in list(sys.modules.keys()):
        if "job_logger" in mod_name:
            sys.modules.pop(mod_name)

    with patch.dict("os.environ", {"LOG_DIR": str(tmp_path)}):
        from src.job_logger import get_logger, _ensure_handler, JobLogger
        import src.job_logger as mod

        mod._HANDLER_INSTALLED = False
        logger = get_logger("test-job-id-001")
        return logger, tmp_path / "app.jsonl"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJobLoggerInfo:
    def test_info_writes_json_line(self, tmp_path: Path) -> None:
        logger, log_file = _make_logger(tmp_path)
        logger.info("upload", filename="video.mp4")
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["job_id"] == "test-job-id-001"
        assert record["step"] == "upload"
        assert record["filename"] == "video.mp4"
        assert record["success"] is None
        assert "timestamp" in record

    def test_success_writes_true(self, tmp_path: Path) -> None:
        logger, log_file = _make_logger(tmp_path)
        logger.success("extract_audio", filename="v.mp4", elapsed=3.5)
        record = json.loads(log_file.read_text().strip().splitlines()[-1])
        assert record["success"] is True
        assert record["elapsed"] == pytest.approx(3.5)
        assert record["step"] == "extract_audio"

    def test_failure_writes_false_and_error(self, tmp_path: Path) -> None:
        logger, log_file = _make_logger(tmp_path)
        logger.failure("transcribe", filename="v.mp4", error="OOM error")
        record = json.loads(log_file.read_text().strip().splitlines()[-1])
        assert record["success"] is False
        assert record["error"] == "OOM error"

    def test_all_fields_present(self, tmp_path: Path) -> None:
        logger, log_file = _make_logger(tmp_path)
        logger.success(
            "burn_subtitles",
            filename="v.mp4",
            video_duration=3600.0,
            elapsed=120.5,
        )
        record = json.loads(log_file.read_text().strip().splitlines()[-1])
        for field in ("timestamp", "job_id", "step", "filename", "video_duration", "elapsed", "success", "error"):
            assert field in record

    def test_japanese_text_in_filename(self, tmp_path: Path) -> None:
        logger, log_file = _make_logger(tmp_path)
        logger.info("upload", filename="日本語動画.mp4")
        record = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert record["filename"] == "日本語動画.mp4"


class TestStepTimer:
    def test_success_case(self, tmp_path: Path) -> None:
        from src.job_logger import StepTimer
        import src.job_logger as mod

        mod._HANDLER_INSTALLED = False
        with patch.dict("os.environ", {"LOG_DIR": str(tmp_path)}):
            mod._HANDLER_INSTALLED = False
            logger, log_file = _make_logger(tmp_path)
            with StepTimer(logger, "extract_audio", filename="v.mp4"):
                time.sleep(0.01)
        lines = log_file.read_text().strip().splitlines()
        # Should have info (start) + success (end) = 2 lines
        assert len(lines) >= 2
        last = json.loads(lines[-1])
        assert last["success"] is True
        assert last["elapsed"] is not None
        assert last["elapsed"] > 0

    def test_failure_case(self, tmp_path: Path) -> None:
        from src.job_logger import StepTimer
        import src.job_logger as mod

        mod._HANDLER_INSTALLED = False
        with patch.dict("os.environ", {"LOG_DIR": str(tmp_path)}):
            logger, log_file = _make_logger(tmp_path)
            with pytest.raises(ValueError):
                with StepTimer(logger, "transcribe", filename="v.mp4"):
                    raise ValueError("boom")
        last = json.loads(log_file.read_text().strip().splitlines()[-1])
        assert last["success"] is False
        assert "boom" in last["error"]
