"""Unit tests for speech-audio chunk extraction in src.ffmpeg_client (Docker mocked)."""

from __future__ import annotations

from pathlib import PurePosixPath
from unittest.mock import patch

import pytest

from src import ffmpeg_client
from src.ffmpeg_client import AudioChunk, extract_speech_audio_chunks, parse_segment_list

CHUNK_DIR = PurePosixPath("/tmp/vt-jobs/job1/speech")


class TestParseSegmentList:
    def test_single_chunk(self) -> None:
        chunks = parse_segment_list("chunk_000.ogg,0.000000,125.340000\n", CHUNK_DIR)
        assert chunks == [AudioChunk(path=CHUNK_DIR / "chunk_000.ogg", start=0.0)]

    def test_multiple_chunks_sorted(self) -> None:
        csv_text = (
            "chunk_001.ogg,14400.010000,20000.000000\n"
            "chunk_000.ogg,0.000000,14400.010000\n"
        )
        chunks = parse_segment_list(csv_text, CHUNK_DIR)
        assert [c.path.name for c in chunks] == ["chunk_000.ogg", "chunk_001.ogg"]
        assert chunks[1].start == pytest.approx(14400.01)

    def test_blank_lines_ignored(self) -> None:
        assert parse_segment_list("\n\n", CHUNK_DIR) == []

    def test_invalid_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Invalid"):
            parse_segment_list("chunk_000.ogg,abc,1\n", CHUNK_DIR)


class TestExtractSpeechAudioChunks:
    def test_runs_ffmpeg_and_reads_list(self) -> None:
        calls: list[list[str]] = []

        def fake_exec(cmd, workdir=None, timeout=None):
            calls.append(cmd)
            if cmd[0] == "cat":
                return 0, "chunk_000.ogg,0.000000,60.000000\n"
            return 0, ""

        with patch.object(ffmpeg_client, "_exec", side_effect=fake_exec):
            chunks = extract_speech_audio_chunks(
                "/tmp/vt-jobs/job1/in.mp4", str(CHUNK_DIR), chunk_seconds=14400
            )

        assert chunks == [AudioChunk(path=CHUNK_DIR / "chunk_000.ogg", start=0.0)]
        ffmpeg_cmd = calls[1]
        assert ffmpeg_cmd[0] == "ffmpeg"
        joined = " ".join(ffmpeg_cmd)
        assert "-ac 1" in joined
        assert "-ar 16000" in joined
        assert "-c:a libopus" in joined
        assert "-b:a 32k" in joined
        assert "-vn" in ffmpeg_cmd
        assert "-segment_time 14400" in joined
        assert ffmpeg_cmd[-1] == str(CHUNK_DIR / "chunk_%03d.ogg")
        assert calls[2] == ["cat", str(CHUNK_DIR / "chunks.csv")]

    def test_ffmpeg_failure_raises(self) -> None:
        def fake_exec(cmd, workdir=None, timeout=None):
            if cmd[0] == "ffmpeg":
                return 1, "Stream map '0:a:0' matches no streams."
            return 0, ""

        with patch.object(ffmpeg_client, "_exec", side_effect=fake_exec):
            with pytest.raises(RuntimeError, match="audio extraction failed"):
                extract_speech_audio_chunks("/in.mp4", str(CHUNK_DIR), chunk_seconds=60)

    def test_empty_list_raises(self) -> None:
        with patch.object(ffmpeg_client, "_exec", return_value=(0, "")):
            with pytest.raises(RuntimeError, match="no audio chunks"):
                extract_speech_audio_chunks("/in.mp4", str(CHUNK_DIR), chunk_seconds=60)
