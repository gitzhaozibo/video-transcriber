"""Job lifecycle management: directory creation, UUID generation, cleanup.

Both the Streamlit host and the FFmpeg container get a per-job temp folder:

* Host:      ``shared/jobs/<job_id>/``       (uploads, extracted audio, SRTs)
* Container: ``/tmp/vt-jobs/<job_id>/``      (created via docker cp / mkdir)

Cleanup removes both sides unless the ``KEEP_TEMP_FILES`` environment
variable is set (used when test-running a job and you want to inspect the
intermediate files afterwards).
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from src.job_logger import JobLogger, StepTimer

# Root of the shared volume on the host (relative or absolute).
from src.ffmpeg_client import (
    HOST_SHARED_DIR,
    container_job_dir,
    remove_container_path,
)

logger = logging.getLogger(__name__)

JOBS_ROOT = HOST_SHARED_DIR / "jobs"


def keep_temp_files() -> bool:
    """Return True when temp folders must be kept (test/debug mode).

    Enabled by setting the ``KEEP_TEMP_FILES`` environment variable to one of
    ``1``, ``true``, ``yes`` or ``on`` (case-insensitive).
    """
    return os.environ.get("KEEP_TEMP_FILES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def new_job_id() -> str:
    """Generate a fresh UUID string for a new job."""
    return str(uuid.uuid4())


def job_dir(job_id: str) -> Path:
    """Return (and create) the host-side working directory for *job_id*."""
    path = JOBS_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_job(job_id: str, job_logger: Optional[JobLogger] = None) -> None:
    """Remove all host-side and container-side files belonging to *job_id*.

    The container-side temp directory is removed too; both sides are kept
    when :func:`keep_temp_files` is True (test mode).

    Parameters
    ----------
    job_id:
        UUID string of the job to clean up.
    job_logger:
        Optional logger to record the cleanup step.
    """
    if keep_temp_files():
        if job_logger is not None:
            job_logger.info("cleanup", extra={"detail": "KEEP_TEMP_FILES set; skipped"})
        logger.info("KEEP_TEMP_FILES set; keeping temp files for job %s", job_id)
        return

    path = JOBS_ROOT / job_id

    def _do_cleanup() -> None:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        # Best-effort removal of the container-side temp directory.
        try:
            remove_container_path(container_job_dir(job_id))
        except Exception as exc:
            logger.warning(
                "Could not remove container temp dir for job %s: %s", job_id, exc
            )

    if job_logger is not None:
        with StepTimer(job_logger, "cleanup"):
            _do_cleanup()
    else:
        _do_cleanup()
