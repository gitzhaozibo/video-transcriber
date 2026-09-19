"""JSON Lines structured logger with daily rotation + PostgreSQL history.

Every log record includes a job ID so that all steps of a single transcription
job can be correlated even if multiple jobs run concurrently.  In addition to
the JSON Lines file, every event is mirrored to the PostgreSQL
``step_events`` table (via :mod:`src.db`) so it is possible to see exactly
where a job is hanging.  When the database is unavailable the file log keeps
working on its own.

Usage example::

    from src.job_logger import get_logger

    job_logger = get_logger("job-uuid-1234")
    job_logger.info("upload", filename="video.mp4", duration=None)
    job_logger.success("extract_audio", filename="video.mp4", duration=3600.0, elapsed=12.3)
    job_logger.failure("transcribe", filename="video.mp4", error="OOM")
"""

from __future__ import annotations

import importlib
import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Setup the underlying Python logger once
# ---------------------------------------------------------------------------

LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "app.jsonl"

_HANDLER_INSTALLED = False


def _ensure_handler() -> None:
    global _HANDLER_INSTALLED
    if _HANDLER_INSTALLED:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(LOG_FILE),
        when="midnight",
        backupCount=30,
        encoding="utf-8",
        utc=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("video_transcriber")
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    _HANDLER_INSTALLED = True


# ---------------------------------------------------------------------------
# JobLogger
# ---------------------------------------------------------------------------

STEPS = frozenset({
    "upload",
    "copy_to_container",
    "probe_duration",
    "extract_audio",
    "copy_from_container",
    "transcribe",
    "translate",
    "burn_subtitles",
    "cleanup",
})


class JobLogger:
    """Structured logger for a single transcription job.

    Parameters
    ----------
    job_id:
        UUID string that identifies this job.
    """

    def __init__(self, job_id: str) -> None:
        _ensure_handler()
        self._job_id = job_id
        self._logger = logging.getLogger("video_transcriber")
        # Import src.db via importlib so tests can swap/patch the module.
        try:
            self._db = importlib.import_module("src.db")
        except Exception:  # pragma: no cover - defensive
            self._db = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def info(
        self,
        step: str,
        *,
        filename: Optional[str] = None,
        video_duration: Optional[float] = None,
        elapsed: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an informational entry (in-progress)."""
        self._emit(
            level="INFO",
            step=step,
            success=None,
            filename=filename,
            video_duration=video_duration,
            elapsed=elapsed,
            error=None,
            extra=extra,
        )

    def success(
        self,
        step: str,
        *,
        filename: Optional[str] = None,
        video_duration: Optional[float] = None,
        elapsed: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a successful step completion."""
        self._emit(
            level="INFO",
            step=step,
            success=True,
            filename=filename,
            video_duration=video_duration,
            elapsed=elapsed,
            error=None,
            extra=extra,
        )

    def failure(
        self,
        step: str,
        *,
        filename: Optional[str] = None,
        video_duration: Optional[float] = None,
        elapsed: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a failed step."""
        self._emit(
            level="ERROR",
            step=step,
            success=False,
            filename=filename,
            video_duration=video_duration,
            elapsed=elapsed,
            error=error,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(
        self,
        *,
        level: str,
        step: str,
        success: Optional[bool],
        filename: Optional[str],
        video_duration: Optional[float],
        elapsed: Optional[float],
        error: Optional[str],
        extra: Optional[dict[str, Any]],
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "job_id": self._job_id,
            "step": step,
            "filename": filename,
            "video_duration": video_duration,
            "elapsed": elapsed,
            "success": success,
            "error": error,
        }
        if extra:
            record.update(extra)

        message = json.dumps(record, ensure_ascii=False)
        if level == "ERROR":
            self._logger.error(message)
        else:
            self._logger.info(message)

        # Mirror the event to PostgreSQL so hangs can be pinpointed from SQL.
        # Failures are swallowed inside src.db (JSONL remains authoritative).
        db = self._db
        if db is not None:
            if success is None:
                db.create_job(self._job_id, filename=filename)
                db.update_job(self._job_id, status="running", current_step=step)
            elif success is True:
                db.update_job(self._job_id, status="running", current_step=step)
            else:
                db.update_job(
                    self._job_id, status="failed", current_step=step, error=error
                )
            db.log_step(
                self._job_id,
                step,
                "running" if success is None else ("success" if success else "failure"),
                filename=filename,
                video_duration=video_duration,
                elapsed=elapsed,
                error=error,
                detail=json.dumps(extra, ensure_ascii=False) if extra else None,
            )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_logger(job_id: str) -> JobLogger:
    """Return a :class:`JobLogger` for *job_id*."""
    return JobLogger(job_id)


# ---------------------------------------------------------------------------
# Context-manager helper for timing a step
# ---------------------------------------------------------------------------


class StepTimer:
    """Context manager that measures elapsed time and logs success/failure.

    Example::

        with StepTimer(job_logger, "extract_audio", filename="v.mp4") as t:
            extract_audio(...)
        # On success: job_logger.success("extract_audio", elapsed=...) is called.
        # On exception: job_logger.failure(..., error=str(exc)) is re-raised.
    """

    def __init__(
        self,
        job_logger: JobLogger,
        step: str,
        *,
        filename: Optional[str] = None,
        video_duration: Optional[float] = None,
    ) -> None:
        self._logger = job_logger
        self._step = step
        self._filename = filename
        self._video_duration = video_duration
        self._start: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.monotonic()
        self._logger.info(
            self._step,
            filename=self._filename,
            video_duration=self._video_duration,
        )
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            self._logger.success(
                self._step,
                filename=self._filename,
                video_duration=self._video_duration,
                elapsed=elapsed,
            )
        else:
            self._logger.failure(
                self._step,
                filename=self._filename,
                video_duration=self._video_duration,
                elapsed=elapsed,
                error=str(exc_val),
            )
        return False  # Do not suppress exceptions
