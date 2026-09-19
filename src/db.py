"""PostgreSQL-backed execution history store.

Stores one :class:`JobRecord` row per job and one :class:`StepRecord` row per
step event so it is possible to see exactly which step a job is currently
hanging on (a ``running`` step with an old ``started_at`` and no
``finished_at``).

All public helpers are *fail-safe*: when the database is unreachable they
return ``None`` instead of raising, so the app keeps working with JSONL file
logging only.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_URL = (
    "******localhost:5432/video_transcriber"
)
DB_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

# ---------------------------------------------------------------------------
# Lazy SQLAlchemy setup
# ---------------------------------------------------------------------------

Base: Any = None
JobRecord: Any = None
StepRecord: Any = None
_engine: Any = None
_Session: Any = None
_available: Optional[bool] = None
_init_lock = threading.Lock()


def _init(force: bool = False) -> bool:
    """Initialise the engine/session factory and create tables.

    Returns True when the database is reachable and ready.
    """
    global Base, JobRecord, StepRecord, _engine, _Session, _available

    with _init_lock:
        if _available is not None and not force:
            return _available
        try:
            from sqlalchemy import (
                Column,
                DateTime,
                Float,
                ForeignKey,
                String,
                Text,
                create_engine,
            )
            from sqlalchemy.orm import declarative_base, sessionmaker

            Base = declarative_base()

            class _JobRecord(Base):
                __tablename__ = "jobs"

                job_id = Column(String(64), primary_key=True)
                filename = Column(String(512), nullable=True)
                status = Column(String(32), nullable=False, default="running")
                current_step = Column(String(64), nullable=True)
                created_at = Column(DateTime, nullable=False)
                updated_at = Column(DateTime, nullable=False)
                finished_at = Column(DateTime, nullable=True)
                error = Column(Text, nullable=True)

            class _StepRecord(Base):
                __tablename__ = "step_events"

                id = Column(String(64), primary_key=True)
                job_id = Column(
                    String(64),
                    ForeignKey("jobs.job_id", ondelete="CASCADE"),
                    nullable=False,
                    index=True,
                )
                step = Column(String(64), nullable=False)
                status = Column(String(32), nullable=False)  # running/success/failure
                filename = Column(String(512), nullable=True)
                video_duration = Column(Float, nullable=True)
                elapsed = Column(Float, nullable=True)
                error = Column(Text, nullable=True)
                detail = Column(Text, nullable=True)
                started_at = Column(DateTime, nullable=False)
                finished_at = Column(DateTime, nullable=True)

            JobRecord = _JobRecord
            StepRecord = _StepRecord

            _engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
            _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
            Base.metadata.create_all(_engine)
            _available = True
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("PostgreSQL unavailable, falling back to JSONL only: %s", exc)
            Base = JobRecord = StepRecord = _engine = _Session = None
            _available = False
        return _available


def is_available() -> bool:
    """Return True if the database has been initialised successfully."""
    return _init() if _available is None else _available


def reset_engine() -> None:
    """Drop cached engine state (used by tests when DATABASE_URL changes)."""
    global _engine, _Session, _available
    with _init_lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
        _engine = None
        _Session = None
        _available = None


def _now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _truncate(value: Optional[str], limit: int = 4000) -> Optional[str]:
    if value is None:
        return None
    return value if len(value) <= limit else value[:limit] + "…"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_job(
    job_id: str,
    filename: Optional[str] = None,
    status: str = "running",
) -> None:
    """Insert a new job row.  No-op when the database is unavailable."""
    if not _init():
        return None
    try:
        now = _now()
        with _Session() as session:
            if session.get(JobRecord, job_id) is None:
                session.add(
                    JobRecord(
                        job_id=job_id,
                        filename=_truncate(filename, 512),
                        status=status,
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DB create_job failed: %s", exc)
    return None


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    current_step: Optional[str] = None,
    error: Optional[str] = None,
    finished: bool = False,
) -> None:
    """Update the job row's status / current step.  No-op without DB."""
    if not _init():
        return None
    try:
        with _Session() as session:
            row = session.get(JobRecord, job_id)
            if row is None:
                row = JobRecord(
                    job_id=job_id, status="running", created_at=_now(), updated_at=_now()
                )
                session.add(row)
            if status is not None:
                row.status = status
            if current_step is not None or status == "running":
                row.current_step = current_step
            if error is not None:
                row.error = _truncate(error)
            row.updated_at = _now()
            if finished:
                row.finished_at = _now()
                row.current_step = None
            session.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DB update_job failed: %s", exc)
    return None


def log_step(
    job_id: str,
    step: str,
    status: str,
    *,
    filename: Optional[str] = None,
    video_duration: Optional[float] = None,
    elapsed: Optional[float] = None,
    error: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """Insert one step-event row.  No-op when the database is unavailable."""
    if not _init():
        return None
    try:
        import uuid as _uuid

        now = _now()
        with _Session() as session:
            session.add(
                StepRecord(
                    id=str(_uuid.uuid4()),
                    job_id=job_id,
                    step=step,
                    status=status,
                    filename=_truncate(filename, 512),
                    video_duration=video_duration,
                    elapsed=elapsed,
                    error=_truncate(error),
                    detail=_truncate(detail),
                    started_at=now,
                    finished_at=None if status == "running" else now,
                )
            )
            session.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DB log_step failed: %s", exc)
    return None


def list_recent_jobs(limit: int = 20) -> Optional[list[dict[str, Any]]]:
    """Return the most recent jobs, or None when the DB is unavailable."""
    if not _init():
        return None
    try:
        with _Session() as session:
            rows = (
                session.query(JobRecord)
                .order_by(JobRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "job_id": r.job_id,
                    "filename": r.filename,
                    "status": r.status,
                    "current_step": r.current_step,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                    "error": r.error,
                }
                for r in rows
            ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DB list_recent_jobs failed: %s", exc)
        return None


def get_steps(job_id: str) -> Optional[list[dict[str, Any]]]:
    """Return all step events for *job_id*, or None when DB unavailable."""
    if not _init():
        return None
    try:
        with _Session() as session:
            rows = (
                session.query(StepRecord)
                .filter(StepRecord.job_id == job_id)
                .order_by(StepRecord.started_at.asc())
                .all()
            )
            return [
                {
                    "step": r.step,
                    "status": r.status,
                    "filename": r.filename,
                    "video_duration": r.video_duration,
                    "elapsed": r.elapsed,
                    "error": r.error,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                }
                for r in rows
            ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DB get_steps failed: %s", exc)
        return None
