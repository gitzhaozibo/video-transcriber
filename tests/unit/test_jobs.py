"""Unit tests for src.jobs (job lifecycle helpers).

Docker SDK and FFmpeg are mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


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
