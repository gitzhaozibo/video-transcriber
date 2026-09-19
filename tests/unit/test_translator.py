"""Unit tests for src.translator (OpenAI API is mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.subtitle import Segment
from src.translator import (
    TranslationError,
    TranslationUnavailableError,
    is_available,
    translate_segments,
    translate_texts,
)


class TestAvailability:
    def test_unavailable_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"streamlit": MagicMock(secrets={})}):
            # st.secrets.get returns None on an empty dict-like mock
            assert is_available() is False

    def test_available_with_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert is_available() is True


class TestTranslateTexts:
    def test_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"streamlit": MagicMock(secrets={})}):
            with pytest.raises(TranslationUnavailableError):
                translate_texts(["hello"])

    def _fake_openai_module(self, content: str) -> MagicMock:
        module = MagicMock()
        client = MagicMock()
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response
        module.OpenAI.return_value = client
        return module

    def test_translation_strips_numbering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake = self._fake_openai_module("1. こんにちは\n2. 世界")
        with patch.dict("sys.modules", {"openai": fake}):
            result = translate_texts(["hello", "world"])
        assert result == ["こんにちは", "世界"]

    def test_line_count_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake = self._fake_openai_module("1. こんにちは")
        with patch.dict("sys.modules", {"openai": fake}):
            with pytest.raises(TranslationError, match="align"):
                translate_texts(["hello", "world"])

    def test_api_error_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake = MagicMock()
        fake.OpenAI.side_effect = Exception("quota exceeded")
        with patch.dict("sys.modules", {"openai": fake}):
            with pytest.raises(TranslationError, match="failed"):
                translate_texts(["hello"])


class TestTranslateSegments:
    def test_empty_input(self) -> None:
        assert translate_segments([]) == []

    def test_timing_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake = MagicMock()
        client = MagicMock()
        message = MagicMock()
        message.content = "1. おはよう\n2. さようなら"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response
        fake.OpenAI.return_value = client

        segments = [Segment(0.0, 1.5, "good morning"), Segment(1.5, 3.0, "goodbye")]
        with patch.dict("sys.modules", {"openai": fake}):
            result = translate_segments(segments)

        assert [r.text for r in result] == ["おはよう", "さようなら"]
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(1.5)
        assert result[1].start == pytest.approx(1.5)
