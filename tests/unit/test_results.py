"""Unit tests for src.results (per-job result JSON) and StepTimer recording."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import results
from src.job_logger import StepTimer
from src.subtitle import Segment


def _result(**overrides) -> dict:
    res = results.new_result(
        "0123456789abcdef",
        filename="会議 録画.mp4",
        size_bytes=1234,
        engine={"name": "azure", "locales": ["ja-JP"]},
    )
    res.update(overrides)
    return res


class TestResultDocument:
    def test_new_result_shape(self) -> None:
        res = _result()
        assert res["schema_version"] == results.SCHEMA_VERSION
        assert res["status"] == "running"
        assert res["input"] == {"filename": "会議 録画.mp4", "size_bytes": 1234,
                                "duration_seconds": None}
        assert res["steps"] == []
        assert res["error"] is None

    def test_segments_payload_accepts_objects_and_dicts(self) -> None:
        rows = results.segments_payload(
            [
                Segment(0.12345, 1.5, " こんにちは ", speaker=1),
                {"start": "2", "end": 3, "text": "世界"},
                {"speaker": float("nan"), "start": 3, "end": 4, "text": "!"},
            ]
        )
        assert rows == [
            {"speaker": 1, "start": 0.123, "end": 1.5, "text": "こんにちは"},
            {"speaker": None, "start": 2.0, "end": 3.0, "text": "世界"},
            {"speaker": None, "start": 3.0, "end": 4.0, "text": "!"},
        ]
        assert list(rows[0]) == ["speaker", "start", "end", "text"]
        # NaN must never reach the JSON file (it is not valid JSON).
        assert "NaN" not in results.to_json({"rows": rows})

    def test_set_transcription(self) -> None:
        res = _result()
        results.set_transcription(res, [Segment(0, 1, "a"), Segment(1, 2, " "), Segment(2, 3, "b")])
        assert res["transcription"]["segment_count"] == 3
        assert res["transcription"]["text"] == "a\nb"

    def test_set_translation_skipped(self) -> None:
        res = _result()
        results.set_translation(res, status="skipped", message="no key")
        assert res["translation"] == {
            "status": "skipped", "model": None, "target_language": None,
            "message": "no key", "segments": None,
        }

    def test_set_error(self) -> None:
        res = _result()
        results.set_error(res, "transcribe", "HTTP 401")
        assert res["status"] == "failed"
        assert res["error"] == {"step": "transcribe", "message": "HTTP 401"}

    def test_filename_is_safe_and_stable(self) -> None:
        res = _result(created_at="2026-11-11T07:30:00+00:00")
        name = results.result_dirname(res)
        assert name.endswith("_会議_録画_01234567")
        assert results.result_dirname(res) == name
        assert results.result_filename(res) == name + ".json"


class TestSaveResult:
    def _read(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_writes_job_folder_and_overwrites(self, tmp_path: Path) -> None:
        res = _result()
        results.set_transcription(res, [Segment(0, 1, "こんにちは", speaker=1)])
        folder1 = results.save_result(res, tmp_path)
        res["status"] = "completed"
        folder2 = results.save_result(res, tmp_path)

        assert folder1 == folder2 == tmp_path / results.result_dirname(res)
        assert [p.name for p in tmp_path.iterdir()] == [folder1.name]
        assert sorted(p.name for p in folder1.iterdir()) == ["result.json", "transcript.json"]
        data = self._read(folder1 / "result.json")
        assert data["status"] == "completed"
        assert data["input"]["filename"] == "会議 録画.mp4"  # not \u-escaped
        assert self._read(folder1 / "transcript.json") == [
            {"speaker": 1, "start": 0.0, "end": 1.0, "text": "こんにちは"}
        ]

    def test_also_writes_job_working_folder(self, tmp_path: Path) -> None:
        res = _result()
        results.set_transcription(res, [Segment(0, 1, "a")])
        job_folder = tmp_path / "shared" / "jobs" / res["job_id"]
        results.save_result(res, tmp_path / "results", job_folder=job_folder)
        assert sorted(p.name for p in job_folder.iterdir()) == ["result.json", "transcript.json"]
        assert self._read(job_folder / "result.json")["job_id"] == res["job_id"]

    def test_translation_and_subtitles_files(self, tmp_path: Path) -> None:
        res = _result()
        results.set_transcription(res, [Segment(0, 1, "Hello")])
        results.set_translation(res, status="completed", model="m", target_language="Japanese",
                                segments=[Segment(0, 1, "こんにちは")])
        res["subtitles"] = {"segments": results.segments_payload([Segment(0, 1, "やあ")])}
        folder = results.save_result(res, tmp_path)
        assert sorted(p.name for p in folder.iterdir()) == [
            "result.json", "subtitles.json", "transcript.json", "translation.json",
        ]
        assert self._read(folder / "translation.json")[0]["text"] == "こんにちは"
        assert self._read(folder / "subtitles.json")[0]["text"] == "やあ"

    def test_skipped_translation_writes_no_file(self, tmp_path: Path) -> None:
        res = _result()
        results.set_transcription(res, [Segment(0, 1, "a")])
        results.set_translation(res, status="skipped", message="no key")
        folder = results.save_result(res, tmp_path)
        assert not (folder / "translation.json").exists()
        assert not list(folder.glob("*.tmp"))

    def test_azure_raw_response_file(self, tmp_path: Path) -> None:
        res = _result()
        results.set_transcription(res, [Segment(0, 1, "a", speaker=1)])
        raw = {"api_version": "2025-10-15",
               "chunks": [{"file": "chunk_000.ogg", "start_seconds": 0.0,
                           "response": {"durationMilliseconds": 1000, "phrases": []}}]}
        results.set_raw_response(res, raw)
        job_folder = tmp_path / "job"
        folder = results.save_result(res, tmp_path / "results", job_folder=job_folder)
        for f in (folder, job_folder):
            assert self._read(f / "azure_response.json") == raw
            data = self._read(f / "result.json")
            assert results.RAW_RESPONSE_KEY not in data   # kept out of result.json
            assert data["engine"]["raw_response_file"] == "azure_response.json"
        assert results.RAW_RESPONSE_KEY not in results.result_document(res)

    def test_no_raw_response_file_without_azure(self, tmp_path: Path) -> None:
        folder = results.save_result(_result(), tmp_path)
        assert not (folder / "azure_response.json").exists()

    def test_creates_directory(self, tmp_path: Path) -> None:
        folder = results.save_result(_result(), tmp_path / "nested" / "results")
        assert (folder / "result.json").exists()


class TestStepTimerRecord:
    def test_records_success(self) -> None:
        steps: list = []
        with StepTimer(MagicMock(), "upload", record=steps) as t:
            pass
        assert steps[0]["step"] == "upload"
        assert steps[0]["success"] is True
        assert steps[0]["error"] is None
        assert t.elapsed is not None

    def test_records_failure_and_reraises(self) -> None:
        steps: list = []
        with pytest.raises(ValueError):
            with StepTimer(MagicMock(), "transcribe", record=steps):
                raise ValueError("boom")
        assert steps == [
            {"step": "transcribe", "elapsed_seconds": steps[0]["elapsed_seconds"],
             "success": False, "error": "boom"}
        ]

    def test_no_record_by_default(self) -> None:
        with StepTimer(MagicMock(), "upload"):
            pass
