"""Unit tests for src.jobs (job lifecycle helpers).

Docker SDK and FFmpeg are mocked.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_container_cleanup():
    """Avoid touching Docker when cleaning up container-side temp dirs."""
    with patch("src.jobs.remove_container_path"):
        yield


class TestNewJobId:
    def test_returns_uuid_string(self) -> None:
        import uuid
        from src.jobs import new_job_id

        job_id = new_job_id()
        # Should be parseable as UUID
        parsed = uuid.UUID(job_id)
        assert str(parsed) == job_id

    def test_unique(self) -> None:
        from src.jobs import new_job_id

        ids = {new_job_id() for _ in range(10)}
        assert len(ids) == 10


class TestJobDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        with patch("src.jobs.JOBS_ROOT", tmp_path / "jobs"):
            from src import jobs

            original_root = jobs.JOBS_ROOT
            jobs.JOBS_ROOT = tmp_path / "jobs"
            try:
                d = jobs.job_dir("test-uuid")
                assert d.exists()
                assert d.is_dir()
            finally:
                jobs.JOBS_ROOT = original_root

    def test_idempotent(self, tmp_path: Path) -> None:
        with patch("src.jobs.JOBS_ROOT", tmp_path / "jobs"):
            from src import jobs

            original_root = jobs.JOBS_ROOT
            jobs.JOBS_ROOT = tmp_path / "jobs"
            try:
                d1 = jobs.job_dir("test-uuid")
                d2 = jobs.job_dir("test-uuid")
                assert d1 == d2
            finally:
                jobs.JOBS_ROOT = original_root


class TestCleanupJob:
    def test_removes_directory(self, tmp_path: Path) -> None:
        from src import jobs

        original_root = jobs.JOBS_ROOT
        jobs.JOBS_ROOT = tmp_path / "jobs"
        try:
            d = jobs.job_dir("cleanup-test")
            (d / "file.txt").write_text("data")
            assert d.exists()

            jobs.cleanup_job("cleanup-test")
            assert not d.exists()
        finally:
            jobs.JOBS_ROOT = original_root

    def test_nonexistent_job_is_safe(self, tmp_path: Path) -> None:
        from src import jobs

        original_root = jobs.JOBS_ROOT
        jobs.JOBS_ROOT = tmp_path / "jobs"
        try:
            # Should not raise
            jobs.cleanup_job("nonexistent-uuid")
        finally:
            jobs.JOBS_ROOT = original_root

    def test_container_cleanup_is_called(self, tmp_path: Path) -> None:
        from src import jobs

        original_root = jobs.JOBS_ROOT
        jobs.JOBS_ROOT = tmp_path / "jobs"
        try:
            jobs.job_dir("ct-cleanup")
            with patch("src.jobs.remove_container_path") as mock_rm:
                jobs.cleanup_job("ct-cleanup")
            mock_rm.assert_called_once()
        finally:
            jobs.JOBS_ROOT = original_root


class TestKeepTempFiles:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KEEP_TEMP_FILES", raising=False)
        from src import jobs

        assert jobs.keep_temp_files() is False

    def test_enabled_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KEEP_TEMP_FILES", "1")
        from src import jobs

        assert jobs.keep_temp_files() is True

    def test_cleanup_keeps_files_in_test_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KEEP_TEMP_FILES", "true")
        from src import jobs

        original_root = jobs.JOBS_ROOT
        jobs.JOBS_ROOT = tmp_path / "jobs"
        try:
            d = jobs.job_dir("keep-me")
            (d / "file.txt").write_text("data")
            jobs.cleanup_job("keep-me")
            assert d.exists()
            assert (d / "file.txt").exists()
        finally:
            jobs.JOBS_ROOT = original_root
