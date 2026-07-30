"""Job lifecycle management: directory creation, UUID generation, cleanup."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from src.job_logger import JobLogger, StepTimer

# Root of the shared volume on the host (relative or absolute).
from src.ffmpeg_client import HOST_SHARED_DIR

JOBS_ROOT = HOST_SHARED_DIR / "jobs"


def new_job_id() -> str:
    """Generate a fresh UUID string for a new job."""
    return str(uuid.uuid4())


def job_dir(job_id: str) -> Path:
    """Return (and create) the working directory for *job_id*."""
    path = JOBS_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_job(job_id: str, job_logger: Optional[JobLogger] = None) -> None:
    """Remove all files belonging to *job_id*.

    Parameters
    ----------
    job_id:
        UUID string of the job to clean up.
    job_logger:
        Optional logger to record the cleanup step.
    """
    path = JOBS_ROOT / job_id
    if not path.exists():
        return

    def _do_cleanup() -> None:
        shutil.rmtree(path, ignore_errors=True)

    if job_logger is not None:
        with StepTimer(job_logger, "cleanup"):
            _do_cleanup()
    else:
        _do_cleanup()
