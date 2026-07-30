"""Unit tests for src.transcriber (faster-whisper is mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.subtitle import Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_yields_segments(self, tmp_path: Path) -> None:
        """transcribe() should yield Segment objects from the model."""
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake audio data")

        fake_segments = [
            _make_fake_segment(0.0, 2.0, "Hello"),
            _make_fake_segment(2.0, 4.5, "World"),
        ]
        fake_info = MagicMock()

        fake_model = MagicMock()
        fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

        with patch("src.transcriber._get_model", return_value=fake_model):
            from src.transcriber import transcribe

            result = list(transcribe(audio))

        assert len(result) == 2
        assert isinstance(result[0], Segment)
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(2.0)
        assert result[0].text == "Hello"
        assert result[1].text == "World"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """transcribe() should raise FileNotFoundError if the file is missing."""
        audio = tmp_path / "nonexistent.m4a"
        with patch("src.transcriber._get_model"):
            from src.transcriber import transcribe

            with pytest.raises(FileNotFoundError):
                list(transcribe(audio))

    def test_model_error_is_wrapped(self, tmp_path: Path) -> None:
        """transcribe() should wrap model errors in RuntimeError."""
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"data")

        fake_model = MagicMock()
        fake_model.transcribe.side_effect = Exception("GPU out of memory")

        with patch("src.transcriber._get_model", return_value=fake_model):
            from src.transcriber import transcribe

            with pytest.raises(RuntimeError, match="transcription failed"):
                list(transcribe(audio))

    def test_language_passed_to_model(self, tmp_path: Path) -> None:
        """transcribe() should pass the language arg to model.transcribe()."""
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"data")

        fake_model = MagicMock()
        fake_model.transcribe.return_value = (iter([]), MagicMock())

        with patch("src.transcriber._get_model", return_value=fake_model):
            from src.transcriber import transcribe

            list(transcribe(audio, language="ja"))

        _call_kwargs = fake_model.transcribe.call_args
        assert _call_kwargs.kwargs.get("language") == "ja" or "ja" in _call_kwargs.args

    def test_empty_transcription(self, tmp_path: Path) -> None:
        """transcribe() with no segments yields nothing."""
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"data")

        fake_model = MagicMock()
        fake_model.transcribe.return_value = (iter([]), MagicMock())

        with patch("src.transcriber._get_model", return_value=fake_model):
            from src.transcriber import transcribe

            result = list(transcribe(audio))
        assert result == []
