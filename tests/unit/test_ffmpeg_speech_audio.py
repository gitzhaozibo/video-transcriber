"""Unit tests for speech-audio extraction in src.ffmpeg_client (Docker mocked)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src import ffmpeg_client
from src.ffmpeg_client import extract_speech_audio

OUTPUT = "/tmp/vt-jobs/job1/in_speech.ogg"


class TestExtractSpeechAudio:
    def test_runs_single_ffmpeg_command(self) -> None:
        calls: list[list[str]] = []

        def fake_exec(cmd, workdir=None, timeout=None):
            calls.append(cmd)
            return 0, ""

        with patch.object(ffmpeg_client, "_exec", side_effect=fake_exec):
            extract_speech_audio("/tmp/vt-jobs/job1/in.mp4", OUTPUT)

        assert len(calls) == 1
        ffmpeg_cmd = calls[0]
        assert ffmpeg_cmd[0] == "ffmpeg"
        joined = " ".join(ffmpeg_cmd)
        assert "-ac 1" in joined
        assert "-ar 16000" in joined
        assert "-c:a libopus" in joined
        assert "-b:a 32k" in joined
        assert "-application voip" in joined
        assert "-vn" in ffmpeg_cmd
        # One output file: no segment muxer.
        assert "segment" not in joined
        assert ffmpeg_cmd[-1] == OUTPUT

    def test_ffmpeg_failure_raises(self) -> None:
        with patch.object(
            ffmpeg_client, "_exec",
            return_value=(1, "Stream map '0:a:0' matches no streams."),
        ):
            with pytest.raises(RuntimeError, match="audio extraction failed"):
                extract_speech_audio("/in.mp4", OUTPUT)
