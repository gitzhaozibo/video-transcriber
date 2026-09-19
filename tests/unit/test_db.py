"""Unit tests for src.db (PostgreSQL execution history).

PostgreSQL is NOT required: the tests run against an in-memory SQLite
database via SQLAlchemy, and cover the fail-safe behaviour when the DB is
unavailable.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture()
def fresh_db(monkeypatch: pytest.MonkeyPatch):
    """Provide a src.db module bound to a fresh in-memory SQLite database."""
    import src.db as db

    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setattr(db, "DB_URL", "sqlite+pysqlite:///:memory:")
    db.reset_engine()
    yield db
    db.reset_engine()


@pytest.fixture()
def broken_db(monkeypatch: pytest.MonkeyPatch):
    """Provide a src.db module whose database connection always fails."""
    import src.db as db

    monkeypatch.setattr(db, "DB_URL", "sqlite+pysqlite:///:memory:")
    db.reset_engine()

    def _failing_create_engine(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("sqlalchemy.create_engine", _failing_create_engine)
    yield db
    db.reset_engine()


class TestJobLifecycle:
    def test_create_and_list(self, fresh_db) -> None:
        fresh_db.create_job("job-1", filename="video.mp4")
        jobs = fresh_db.list_recent_jobs()
        assert jobs is not None
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "job-1"
        assert jobs[0]["filename"] == "video.mp4"
        assert jobs[0]["status"] == "running"

    def test_update_status_and_step(self, fresh_db) -> None:
        fresh_db.create_job("job-2")
        fresh_db.update_job("job-2", status="running", current_step="transcribe")
        job = fresh_db.list_recent_jobs()[0]
        assert job["current_step"] == "transcribe"

    def test_finish_clears_current_step(self, fresh_db) -> None:
        fresh_db.create_job("job-3")
        fresh_db.update_job("job-3", current_step="burn_subtitles")
        fresh_db.update_job("job-3", status="completed", finished=True)
        job = fresh_db.list_recent_jobs()[0]
        assert job["status"] == "completed"
        assert job["current_step"] is None

    def test_update_unknown_job_creates_row(self, fresh_db) -> None:
        fresh_db.update_job("job-4", current_step="upload")
        jobs = fresh_db.list_recent_jobs()
        assert jobs is not None and jobs[0]["job_id"] == "job-4"

    def test_long_error_is_truncated(self, fresh_db) -> None:
        fresh_db.create_job("job-5")
        fresh_db.update_job("job-5", status="failed", error="x" * 10000)
        job = fresh_db.list_recent_jobs()[0]
        assert len(job["error"]) <= 4100


class TestStepEvents:
    def test_running_step_has_no_finished_at(self, fresh_db) -> None:
        fresh_db.create_job("job-10")
        fresh_db.log_step("job-10", "transcribe", "running")
        steps = fresh_db.get_steps("job-10")
        assert steps is not None and len(steps) == 1
        assert steps[0]["status"] == "running"
        assert steps[0]["finished_at"] is None

    def test_success_step_records_elapsed(self, fresh_db) -> None:
        fresh_db.create_job("job-11")
        fresh_db.log_step("job-11", "extract_audio", "success", elapsed=1.5)
        steps = fresh_db.get_steps("job-11")
        assert steps[0]["elapsed"] == pytest.approx(1.5)
        assert steps[0]["finished_at"] is not None

    def test_failure_step_records_error(self, fresh_db) -> None:
        fresh_db.create_job("job-12")
        fresh_db.log_step("job-12", "burn_subtitles", "failure", error="exit=1")
        steps = fresh_db.get_steps("job-12")
        assert steps[0]["error"] == "exit=1"

    def test_steps_are_ordered(self, fresh_db) -> None:
        fresh_db.create_job("job-13")
        for step in ("upload", "probe_duration", "transcribe"):
            fresh_db.log_step("job-13", step, "running")
        steps = fresh_db.get_steps("job-13")
        assert [s["step"] for s in steps] == ["upload", "probe_duration", "transcribe"]

    def test_get_steps_for_unknown_job_is_empty(self, fresh_db) -> None:
        assert fresh_db.get_steps("nope") == []


class TestFailSafe:
    def test_unavailable_db_returns_none(self, broken_db) -> None:
        assert broken_db.is_available() is False
        assert broken_db.list_recent_jobs() is None
        assert broken_db.get_steps("job-x") is None

    def test_unavailable_db_writes_are_noops(self, broken_db) -> None:
        # Should not raise even though the DB is down.
        broken_db.create_job("job-x", filename="v.mp4")
        broken_db.update_job("job-x", status="failed", error="boom")
        broken_db.log_step("job-x", "upload", "failure", error="boom")


class TestAvailability:
    def test_is_available_true(self, fresh_db) -> None:
        assert fresh_db.is_available() is True
