"""Unit tests for src.azure_transcriber (HTTP calls are mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src import azure_transcriber as az
from src.subtitle import Segment

_AZURE_ENV_VARS = (
    "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_ENDPOINT",
    "AZURE_SPEECH_REGION",
    "AZURE_SPEECH_LOCALES",
    "AZURE_SPEECH_API_VERSION",
    "AZURE_MAX_SEGMENT_SECONDS",
    "AZURE_MAX_CHUNK_SECONDS",
    "AZURE_REQUEST_TIMEOUT",
    "AZURE_DIARIZATION",
    "AZURE_MAX_SPEAKERS",
    "AZURE_PHRASE_LIST",
    "AZURE_PHRASE_LIST_WEIGHT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AZURE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(az, "_sleep", lambda _s: None)


def _config(**overrides) -> az.AzureSpeechConfig:
    values = {"key": "test-key", "endpoint": "https://example.cognitiveservices.azure.com"}
    values.update(overrides)
    return az.AzureSpeechConfig(**values)


def _response(status: int, body=None, headers=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.text = json.dumps(body) if body is not None else ""
    resp.json.return_value = body
    return resp


def _phrase(offset_ms: int, duration_ms: int, text: str, words=None) -> dict:
    phrase = {"offsetMilliseconds": offset_ms, "durationMilliseconds": duration_ms, "text": text}
    if words is not None:
        phrase["words"] = [
            {"text": t, "offsetMilliseconds": o, "durationMilliseconds": d}
            for t, o, d in words
        ]
    return phrase


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_key(self) -> None:
        with pytest.raises(az.AzureTranscriptionConfigError, match="AZURE_SPEECH_KEY"):
            az.load_config()

    def test_missing_endpoint_and_region(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        with pytest.raises(az.AzureTranscriptionConfigError, match="ENDPOINT"):
            az.load_config()

    def test_region_builds_endpoint(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        cfg = az.load_config()
        assert cfg.url == (
            "https://japaneast.api.cognitive.microsoft.com/speechtotext/"
            f"transcriptions:transcribe?api-version={az.DEFAULT_API_VERSION}"
        )

    def test_endpoint_wins_and_trailing_slash_stripped(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", "https://my.cognitiveservices.azure.com/")
        cfg = az.load_config()
        assert cfg.endpoint == "https://my.cognitiveservices.azure.com"

    def test_default_locales(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        assert az.load_config().locales == ("ja-JP", "en-US", "zh-CN", "de-DE")

    @pytest.mark.parametrize(
        "name, value",
        [
            ("AZURE_SPEECH_LOCALES", "multilingual"),
            ("AZURE_PHRASE_LIST_WEIGHT", "heavy"),
            ("AZURE_MAX_CHUNK_SECONDS", "0"),
        ],
    )
    def test_removed_settings_are_ignored(self, monkeypatch, name, value) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        monkeypatch.setenv(name, value)
        cfg = az.load_config()
        assert cfg.locales == ("ja-JP", "en-US", "zh-CN", "de-DE")
        assert "biasingWeight" not in json.dumps(az.build_definition(cfg))

    def test_diarization_defaults(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        cfg = az.load_config()
        assert cfg.diarization is True
        assert cfg.max_speakers == 4

    def test_diarization_settings(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        monkeypatch.setenv("AZURE_DIARIZATION", "false")
        monkeypatch.setenv("AZURE_MAX_SPEAKERS", "10")
        cfg = az.load_config()
        assert cfg.diarization is False
        assert cfg.max_speakers == 10

    @pytest.mark.parametrize(
        "name, value",
        [("AZURE_DIARIZATION", "maybe"), ("AZURE_MAX_SPEAKERS", "1"), ("AZURE_MAX_SPEAKERS", "36")],
    )
    def test_invalid_diarization_settings(self, monkeypatch, name, value) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        monkeypatch.setenv(name, value)
        with pytest.raises(az.AzureTranscriptionConfigError, match=name):
            az.load_config()

# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestLanguageAndPhraseList:
    @pytest.fixture(autouse=True)
    def _creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")

    def test_fixed_candidate_locales(self) -> None:
        cfg = az.load_config()
        assert cfg.language_label == "ja-JP, en-US, zh-CN, de-DE"
        assert az.build_definition(cfg)["locales"] == ["ja-JP", "en-US", "zh-CN", "de-DE"]

    def test_no_phrase_list_by_default(self) -> None:
        cfg = az.load_config()
        assert cfg.phrase_list == ()
        assert "phraseList" not in az.build_definition(cfg)

    def test_phrase_list_parsed_and_sent(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_PHRASE_LIST", "Contoso, 山田太郎\nゲートキーパー,,Contoso")
        cfg = az.load_config()
        assert cfg.phrase_list == ("Contoso", "山田太郎", "ゲートキーパー")
        assert az.build_definition(cfg)["phraseList"] == {
            "phrases": ["Contoso", "山田太郎", "ゲートキーパー"]
        }

    def test_phrase_list_in_request(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"x")
        with patch.object(az.requests, "post",
                          return_value=_response(200, {"phrases": []})) as post:
            az.transcribe_file(audio, config=_config(phrase_list=("Contoso",)))
        definition = json.loads(post.call_args.kwargs["files"]["definition"][1])
        assert definition["phraseList"] == {"phrases": ["Contoso"]}


class TestLocaleSummary:
    def test_orders_by_speaking_time(self) -> None:
        def p(ms: int, locale, text="x"):
            phrase = {"offsetMilliseconds": 0, "durationMilliseconds": ms, "text": text}
            if locale:
                phrase["locale"] = locale
            return phrase

        response = {"phrases": [
            p(2000, "ja-JP"), p(1500, "en-US"), p(900, None),
            p(3000, "ja-JP"), p(4000, "en-US", text=" "),
        ]}
        assert az.locale_summary(response) == {
            "locale": "ja-JP",
            "locales": [
                {"locale": "ja-JP", "seconds": 5.0, "phrases": 2},
                {"locale": "en-US", "seconds": 1.5, "phrases": 1},
                {"locale": "unknown", "seconds": 0.9, "phrases": 1},
            ],
        }

    def test_empty(self) -> None:
        assert az.locale_summary({}) == {"locale": None, "locales": []}


class TestPhrasesToSegments:
    def test_basic_conversion_and_sorting(self) -> None:
        result = {
            "phrases": [
                _phrase(3000, 1500, "二つ目"),
                _phrase(960, 640, " Good afternoon. "),
            ]
        }
        segs = az.phrases_to_segments(result)
        assert segs == [
            Segment(start=0.96, end=pytest.approx(1.6), text="Good afternoon."),
            Segment(start=3.0, end=4.5, text="二つ目"),
        ]

    def test_empty_text_skipped(self) -> None:
        result = {"phrases": [_phrase(0, 1000, "  ")]}
        assert az.phrases_to_segments(result) == []

    def test_no_phrases(self) -> None:
        assert az.phrases_to_segments({}) == []

    def test_long_phrase_is_kept_whole(self) -> None:
        text = "今日は晴れです。明日は雨でしょう。"
        words = [("今日は晴れです", 0, 4500), ("明日は雨でしょう", 6000, 5000)]
        phrase = _phrase(0, 30000, text, words)
        phrase["speaker"] = 2
        segs = az.phrases_to_segments({"phrases": [phrase]})
        assert segs == [Segment(start=0.0, end=30.0, text=text, speaker=2)]

    def test_no_speaker_is_none(self) -> None:
        segs = az.phrases_to_segments({"phrases": [_phrase(0, 1000, "x")]})
        assert segs[0].speaker is None


# ---------------------------------------------------------------------------
# HTTP / transcribe_file
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    def _audio(self, tmp_path: Path) -> Path:
        audio = tmp_path / "input_speech.ogg"
        audio.write_bytes(b"fake audio")
        return audio

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            az.transcribe_file(tmp_path / "missing.ogg", config=_config())

    def test_request_contents(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        body = {"phrases": [_phrase(0, 1000, "こんにちは")]}
        with patch.object(az.requests, "post", return_value=_response(200, body)) as post:
            segs, raw = az.transcribe_file(audio, config=_config())

        assert segs == [Segment(start=0.0, end=1.0, text="こんにちは")]
        assert raw == body
        assert post.call_count == 1
        args, kwargs = post.call_args
        assert args[0].startswith(
            "https://example.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe"
        )
        assert kwargs["headers"] == {"Ocp-Apim-Subscription-Key": "test-key"}
        name, _fh, mime = kwargs["files"]["audio"]
        assert (name, mime) == ("input_speech.ogg", "audio/ogg")
        definition = json.loads(kwargs["files"]["definition"][1])
        assert definition == {
            "locales": ["ja-JP", "en-US", "zh-CN", "de-DE"],
            "profanityFilterMode": "None",
            "diarization": {"enabled": True, "maxSpeakers": 4},
        }

    def test_diarization_disabled_omits_option(self) -> None:
        assert "diarization" not in az.build_definition(_config(diarization=False))

    def test_speakers_in_result(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        body = {"phrases": [
            {"offsetMilliseconds": 0, "durationMilliseconds": 1000, "text": "はい", "speaker": 1},
            {"offsetMilliseconds": 1200, "durationMilliseconds": 800, "text": "どうも", "speaker": 2},
        ]}
        with patch.object(az.requests, "post", return_value=_response(200, body)):
            segs, _raw = az.transcribe_file(audio, config=_config())
        assert [(s.speaker, s.text) for s in segs] == [(1, "はい"), (2, "どうも")]

    def test_retries_on_429_then_succeeds(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        responses = [
            _response(429, {"error": "busy"}, {"Retry-After": "2"}),
            _response(200, {"phrases": []}),
        ]
        with patch.object(az.requests, "post", side_effect=responses) as post:
            assert az.transcribe_file(audio, config=_config()) == ([], {"phrases": []})
        assert post.call_count == 2

    def test_retries_on_connection_error(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        effects = [requests.ConnectionError("reset"), _response(200, {"phrases": []})]
        with patch.object(az.requests, "post", side_effect=effects) as post:
            az.transcribe_file(audio, config=_config())
        assert post.call_count == 2

    def test_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        with patch.object(
            az.requests, "post", return_value=_response(503, {"error": "down"})
        ) as post:
            with pytest.raises(az.AzureTranscriptionError, match="HTTP 503"):
                az.transcribe_file(audio, config=_config())
        assert post.call_count == az.MAX_ATTEMPTS

    def test_client_error_not_retried(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        with patch.object(
            az.requests, "post", return_value=_response(401, {"error": "bad key"})
        ) as post:
            with pytest.raises(az.AzureTranscriptionError, match="HTTP 401"):
                az.transcribe_file(audio, config=_config())
        assert post.call_count == 1

    def test_timeout_is_wrapped(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        with patch.object(az.requests, "post", side_effect=requests.ReadTimeout("slow")):
            with pytest.raises(az.AzureTranscriptionError, match="request failed"):
                az.transcribe_file(audio, config=_config())


    def test_file_too_large_is_not_sent(self, tmp_path: Path, monkeypatch) -> None:
        audio = self._audio(tmp_path)
        monkeypatch.setattr(az, "API_MAX_FILE_BYTES", 5)
        with patch.object(az.requests, "post") as post:
            with pytest.raises(az.AzureTranscriptionError, match="too large"):
                az.transcribe_file(audio, config=_config())
        post.assert_not_called()

    def test_diarization_sends_only_max_speakers(self) -> None:
        assert az.build_definition(_config(max_speakers=6))["diarization"] == {
            "enabled": True, "maxSpeakers": 6,
        }


class TestResponseDocument:
    def test_wraps_response_with_settings(self) -> None:
        body = {"durationMilliseconds": 1500, "phrases": [_phrase(0, 1000, "x")]}
        doc = az.response_document(_config(), "input_speech.ogg", body)
        assert doc == {
            "api": "fast transcription",
            "api_version": az.DEFAULT_API_VERSION,
            "endpoint": "https://example.cognitiveservices.azure.com",
            "definition": az.build_definition(_config()),
            "file": "input_speech.ogg",
            "response": body,
        }
        assert "test-key" not in json.dumps(doc)
