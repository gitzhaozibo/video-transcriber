"""Per-job result files (JSON).

Each job gets a folder ``RESULTS_DIR/<YYYYmmdd-HHMMSS>_<stem>_<id8>/`` (default
``results/``) that is rewritten as the job progresses (transcribed →
completed / failed).  The same files are also written to the job's working
folder ``shared/jobs/<job_id>/``; that copy is removed with the job's other
temp files on cleanup, while ``results/`` is never cleaned up.

Files in each folder::

    result.json       the whole result document (schema below)
    transcript.json   transcription segments  [{"speaker", "start", "end", "text"}, ...]
    translation.json  translated segments     (only when translation succeeded)
    subtitles.json    segments burned into the video (only after burning)
    azure_response.json  raw Azure fast transcription responses
                      (``response.json()`` per audio chunk; Azure jobs only)

Schema of ``result.json`` (``schema_version`` 1)::

    {
      "schema_version": 1,
      "job_id": "...",
      "status": "running" | "transcribed" | "completed" | "failed",
      "created_at": "...", "updated_at": "...",      # ISO 8601, UTC
      "input": {"filename": "...", "size_bytes": 123, "duration_seconds": 61.2},
      "engine": {"name": "whisper" | "azure", ...engine settings...,
                 "raw_response_file": "azure_response.json"},   # Azure only
      "steps": [{"step": "upload", "elapsed_seconds": 1.2,
                 "success": true, "error": null}, ...],
      "transcription": {"segment_count": 2, "text": "...",
                        "language": {"locale": "ja-JP",          # Azure only
                                     "locales": [{"locale": "ja-JP",
                                                  "seconds": 52.1,
                                                  "phrases": 12}]} | null,
                        "segments": [{"speaker": 1 | null, "start": 0.0,
                                      "end": 2.5, "text": "..."}]},
      "translation": {"status": "completed" | "skipped" | "failed",
                      "model": "...", "target_language": "...",
                      "message": null, "segments": [...]} | null,
      "subtitles": {"segments": [...]} | null,         # rows burned into the video
      "output": {"filename": "..._subtitled.mp4"} | null,
      "error": {"step": "...", "message": "..."} | null
    }
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from src.subtitle import Segment, parse_speaker

SCHEMA_VERSION = 1
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def segments_payload(segments: Sequence[Segment | dict]) -> list[dict[str, Any]]:
    """Normalise segments (objects or row dicts) to ``speaker/start/end/text`` rows.

    ``speaker`` is an integer for engines with diarization (Azure) and ``None``
    otherwise (Whisper).
    """
    rows = []
    for seg in segments:
        if isinstance(seg, Segment):
            speaker, start, end, text = seg.speaker, seg.start, seg.end, seg.text
        else:
            speaker = parse_speaker(seg.get("speaker"))
            start, end, text = seg["start"], seg["end"], seg["text"]
        rows.append(
            {
                "speaker": speaker,
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "text": str(text).strip(),
            }
        )
    return rows


def new_result(
    job_id: str,
    *,
    filename: str,
    size_bytes: Optional[int],
    engine: dict[str, Any],
) -> dict[str, Any]:
    """Return a fresh result document for a job that is starting."""
    now = _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "input": {"filename": filename, "size_bytes": size_bytes, "duration_seconds": None},
        "engine": engine,
        "steps": [],
        "transcription": None,
        "translation": None,
        "subtitles": None,
        "output": None,
        "error": None,
    }


def set_transcription(
    result: dict[str, Any],
    segments: Sequence[Segment | dict],
    *,
    language: Optional[dict[str, Any]] = None,
) -> None:
    """Store the transcription; *language* is the detected-locale summary (Azure)."""
    rows = segments_payload(segments)
    result["transcription"] = {
        "segment_count": len(rows),
        "text": "\n".join(r["text"] for r in rows if r["text"]),
        "language": language,
        "segments": rows,
    }


def set_translation(
    result: dict[str, Any],
    *,
    status: str,
    model: Optional[str] = None,
    target_language: Optional[str] = None,
    segments: Optional[Sequence[Segment | dict]] = None,
    message: Optional[str] = None,
) -> None:
    result["translation"] = {
        "status": status,
        "model": model,
        "target_language": target_language,
        "message": message,
        "segments": segments_payload(segments) if segments is not None else None,
    }


def set_error(result: dict[str, Any], step: str, message: str) -> None:
    result["status"] = "failed"
    result["error"] = {"step": step, "message": message}


def set_raw_response(result: dict[str, Any], document: dict[str, Any]) -> None:
    """Attach the raw Azure response document (saved as azure_response.json)."""
    result[RAW_RESPONSE_KEY] = document
    result["engine"]["raw_response_file"] = AZURE_RESPONSE_FILE


def result_document(result: dict[str, Any]) -> dict[str, Any]:
    """The content of result.json: *result* without the raw Azure responses."""
    return {k: v for k, v in result.items() if k != RAW_RESPONSE_KEY}


RESULT_FILE = "result.json"
TRANSCRIPT_FILE = "transcript.json"
TRANSLATION_FILE = "translation.json"
SUBTITLES_FILE = "subtitles.json"
AZURE_RESPONSE_FILE = "azure_response.json"

# Key of the in-memory result dict holding the raw Azure responses.  It is
# written to AZURE_RESPONSE_FILE and left out of result.json (it can be large).
RAW_RESPONSE_KEY = "_azure_response"


def result_dirname(result: dict[str, Any]) -> str:
    """``<YYYYmmdd-HHMMSS>_<input stem>_<job id prefix>`` (stable per job)."""
    created = datetime.fromisoformat(result["created_at"]).astimezone()
    stem = Path(result["input"]["filename"] or "job").stem
    safe_stem = re.sub(r"[^\w.-]+", "_", stem).strip("._")[:60] or "job"
    return f"{created:%Y%m%d-%H%M%S}_{safe_stem}_{result['job_id'][:8]}"


def result_filename(result: dict[str, Any]) -> str:
    """File name offered for downloading the whole result document."""
    return result_dirname(result) + ".json"


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def result_files(result: dict[str, Any]) -> dict[str, Any]:
    """Return ``{file name: JSON data}`` for every file of a job folder.

    ``None`` means the file does not apply (yet) and should not exist.
    """
    translation = result.get("translation") or {}
    subtitles = result.get("subtitles") or {}
    transcription = result.get("transcription") or {}
    return {
        RESULT_FILE: result_document(result),
        TRANSCRIPT_FILE: transcription.get("segments"),
        TRANSLATION_FILE: translation.get("segments"),
        SUBTITLES_FILE: subtitles.get("segments"),
        AZURE_RESPONSE_FILE: result.get(RAW_RESPONSE_KEY),
    }


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(to_json(data) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_job_folder(result: dict[str, Any], folder: Path) -> Path:
    """Write all result files into *folder* (atomically, file by file)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in result_files(result).items():
        path = folder / name
        if data is None:
            path.unlink(missing_ok=True)
        else:
            _write_json(path, data)
    return folder


def save_result(
    result: dict[str, Any],
    results_dir: Optional[Path] = None,
    *,
    job_folder: Optional[Path] = None,
) -> Path:
    """Save the job's files to ``results_dir/<job folder>/`` and *job_folder*.

    Returns the folder under *results_dir* (default :data:`RESULTS_DIR`).
    *job_folder* is the job's working folder (``shared/jobs/<job_id>/``); it
    is written as well when given.
    """
    result["updated_at"] = _now()
    root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    folder = write_job_folder(result, root / result_dirname(result))
    if job_folder is not None:
        write_job_folder(result, Path(job_folder))
    return folder
