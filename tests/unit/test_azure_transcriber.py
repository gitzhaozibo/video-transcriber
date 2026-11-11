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

    def test_custom_locales(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        monkeypatch.setenv("AZURE_SPEECH_LOCALES", " ja-JP , en-US ,")
        assert az.load_config().locales == ("ja-JP", "en-US")

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

    @pytest.mark.parametrize("value", ["0", "18000", "abc"])
    def test_invalid_chunk_seconds(self, monkeypatch, value) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")
        monkeypatch.setenv("AZURE_MAX_CHUNK_SECONDS", value)
        with pytest.raises(az.AzureTranscriptionConfigError):
            az.load_config()


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestLanguageAndPhraseList:
    @pytest.fixture(autouse=True)
    def _creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "k")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")

    def test_default_is_language_identification(self) -> None:
        cfg = az.load_config()
        assert cfg.language_mode == "identification"
        assert not cfg.multilingual
        assert cfg.language_label == "ja-JP, en-US, zh-CN, de-DE"
        assert az.build_definition(cfg)["locales"] == ["ja-JP", "en-US", "zh-CN", "de-DE"]

    @pytest.mark.parametrize("value", ["multilingual", "MULTILINGUAL", " multilingual "])
    def test_multilingual_sends_empty_locales(self, monkeypatch, value) -> None:
        monkeypatch.setenv("AZURE_SPEECH_LOCALES", value)
        cfg = az.load_config()
        assert cfg.locales == ()
        assert cfg.multilingual
        assert cfg.language_mode == "multilingual"
        assert "multilingual" in cfg.language_label
        definition = az.build_definition(cfg)
        assert definition["locales"] == []
        assert definition["diarization"]["enabled"] is True  # still combined

    def test_single_locale_is_fixed(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_LOCALES", "ja-JP")
        assert az.load_config().language_mode == "fixed"

    def test_multilingual_cannot_be_mixed(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_LOCALES", "multilingual,ja-JP")
        with pytest.raises(az.AzureTranscriptionConfigError, match="multilingual"):
            az.load_config()

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

    def test_phrase_list_weight(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_PHRASE_LIST", "Contoso")
        monkeypatch.setenv("AZURE_PHRASE_LIST_WEIGHT", "1.5")
        cfg = az.load_config()
        assert az.build_definition(cfg)["phraseList"] == {
            "phrases": ["Contoso"], "biasingWeight": 1.5,
        }

    @pytest.mark.parametrize("value", ["2.5", "-0.1", "heavy"])
    def test_invalid_phrase_list_weight(self, monkeypatch, value) -> None:
        monkeypatch.setenv("AZURE_PHRASE_LIST_WEIGHT", value)
        with pytest.raises(az.AzureTranscriptionConfigError, match="AZURE_PHRASE_LIST_WEIGHT"):
            az.load_config()

    def test_phrase_list_in_request(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"x")
        with patch.object(az.requests, "post",
                          return_value=_response(200, {"phrases": []})) as post:
            az.transcribe_file(audio, config=_config(phrase_list=("Contoso",)))
        definition = json.loads(post.call_args.kwargs["files"]["definition"][1])
        assert definition["phraseList"] == {"phrases": ["Contoso"]}


class TestLocaleSummary:
    def test_orders_by_speaking_time_across_chunks(self) -> None:
        def p(ms: int, locale, text="x"):
            phrase = {"offsetMilliseconds": 0, "durationMilliseconds": ms, "text": text}
            if locale:
                phrase["locale"] = locale
            return phrase

        responses = [
            {"response": {"phrases": [p(2000, "ja-JP"), p(1500, "en-US"), p(900, None)]}},
            {"response": {"phrases": [p(3000, "ja-JP"), p(4000, "en-US", text=" ")]}},
            {"response": {}},
        ]
        assert az.locale_summary(responses) == {
            "locale": "ja-JP",
            "locales": [
                {"locale": "ja-JP", "seconds": 5.0, "phrases": 2},
                {"locale": "en-US", "seconds": 1.5, "phrases": 1},
                {"locale": "unknown", "seconds": 0.9, "phrases": 1},
            ],
        }

    def test_empty(self) -> None:
        assert az.locale_summary([]) == {"locale": None, "locales": []}


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

    def test_offset_applied(self) -> None:
        result = {"phrases": [_phrase(1000, 2000, "hello")]}
        segs = az.phrases_to_segments(result, offset=14400.0)
        assert segs[0].start == pytest.approx(14401.0)
        assert segs[0].end == pytest.approx(14403.0)

    def test_empty_text_skipped(self) -> None:
        result = {"phrases": [_phrase(0, 1000, "  ")]}
        assert az.phrases_to_segments(result) == []

    def test_no_phrases(self) -> None:
        assert az.phrases_to_segments({}) == []

    def test_long_phrase_split_preserves_punctuation(self) -> None:
        text = "今日は晴れです。明日は雨でしょう。"
        words = [
            ("今日", 0, 1000),
            ("は", 1000, 500),
            ("晴れ", 1500, 2000),
            ("です", 3500, 1000),
            ("明日", 6000, 1000),
            ("は", 7000, 500),
            ("雨", 7500, 1500),
            ("でしょう", 9000, 2000),
        ]
        result = {"phrases": [_phrase(0, 11000, text, words)]}
        segs = az.phrases_to_segments(result, max_segment_seconds=5.0)
        assert [s.text for s in segs] == ["今日は晴れです。", "明日は雨でしょう。"]
        assert segs[0].start == 0.0
        assert segs[0].end == pytest.approx(4.5)
        assert segs[1].start == pytest.approx(6.0)
        assert segs[1].end == pytest.approx(11.0)

    def test_long_phrase_with_spaces(self) -> None:
        text = "Hello there, how are you today?"
        words = [
            ("Hello", 0, 2000),
            ("there,", 2000, 2000),
            ("how", 5000, 2000),
            ("are", 7000, 1000),
            ("you", 8000, 1000),
            ("today?", 9000, 2000),
        ]
        result = {"phrases": [_phrase(0, 11000, text, words)]}
        segs = az.phrases_to_segments(result, max_segment_seconds=6.0)
        assert [s.text for s in segs] == ["Hello there,", "how are you today?"]

    def test_unalignable_words_keep_whole_phrase(self) -> None:
        words = [("foo", 0, 6000), ("bar", 6000, 6000)]
        result = {"phrases": [_phrase(0, 12000, "全く別のテキスト", words)]}
        segs = az.phrases_to_segments(result, max_segment_seconds=5.0)
        assert len(segs) == 1
        assert segs[0].text == "全く別のテキスト"

    def test_speaker_kept_on_split_pieces(self) -> None:
        text = "今日は晴れです。明日は雨でしょう。"
        words = [("今日は晴れです", 0, 4500), ("明日は雨でしょう", 6000, 5000)]
        phrase = _phrase(0, 11000, text, words)
        phrase["speaker"] = 2
        segs = az.phrases_to_segments({"phrases": [phrase]}, max_segment_seconds=5.0)
        assert len(segs) == 2
        assert all(s.speaker == 2 for s in segs)

    def test_no_speaker_is_none(self) -> None:
        segs = az.phrases_to_segments({"phrases": [_phrase(0, 1000, "x")]})
        assert segs[0].speaker is None

    def test_split_disabled_with_zero(self) -> None:
        words = [("a", 0, 6000), ("b", 6000, 6000)]
        result = {"phrases": [_phrase(0, 12000, "a b", words)]}
        segs = az.phrases_to_segments(result, max_segment_seconds=0)
        assert len(segs) == 1


# ---------------------------------------------------------------------------
# HTTP / transcribe_file
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    def _audio(self, tmp_path: Path) -> Path:
        audio = tmp_path / "chunk_000.ogg"
        audio.write_bytes(b"fake audio")
        return audio

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            az.transcribe_file(tmp_path / "missing.ogg", config=_config())

    def test_request_contents(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        body = {"phrases": [_phrase(0, 1000, "こんにちは")]}
        with patch.object(az.requests, "post", return_value=_response(200, body)) as post:
            segs = az.transcribe_file(audio, config=_config())

        assert segs == [Segment(start=0.0, end=1.0, text="こんにちは")]
        args, kwargs = post.call_args
        assert args[0].startswith(
            "https://example.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe"
        )
        assert kwargs["headers"] == {"Ocp-Apim-Subscription-Key": "test-key"}
        name, _fh, mime = kwargs["files"]["audio"]
        assert (name, mime) == ("chunk_000.ogg", "audio/ogg")
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
            segs = az.transcribe_file(audio, config=_config())
        assert [(s.speaker, s.text) for s in segs] == [(1, "はい"), (2, "どうも")]

    def test_retries_on_429_then_succeeds(self, tmp_path: Path) -> None:
        audio = self._audio(tmp_path)
        responses = [
            _response(429, {"error": "busy"}, {"Retry-After": "2"}),
            _response(200, {"phrases": []}),
        ]
        with patch.object(az.requests, "post", side_effect=responses) as post:
            assert az.transcribe_file(audio, config=_config()) == []
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


class TestTranscribeChunks:
    def test_chunks_are_offset_and_ordered(self, tmp_path: Path) -> None:
        a = tmp_path / "chunk_000.ogg"
        b = tmp_path / "chunk_001.ogg"
        a.write_bytes(b"a")
        b.write_bytes(b"b")

        def fake_post(url, **kwargs):
            name = kwargs["files"]["audio"][0]
            text = "first" if name == "chunk_000.ogg" else "second"
            return _response(200, {"phrases": [_phrase(500, 1000, text)]})

        with patch.object(az.requests, "post", side_effect=fake_post):
            segs = az.transcribe_chunks([(b, 14400.0), (a, 0.0)], config=_config())

        assert [s.text for s in segs] == ["first", "second"]
        assert segs[0].start == pytest.approx(0.5)
        assert segs[1].start == pytest.approx(14400.5)

    def test_collects_raw_responses(self, tmp_path: Path) -> None:
        a = tmp_path / "chunk_000.ogg"
        b = tmp_path / "chunk_001.ogg"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        bodies = {
            "chunk_000.ogg": {"durationMilliseconds": 1500, "phrases": [_phrase(0, 1000, "x")]},
            "chunk_001.ogg": {"durationMilliseconds": 900, "phrases": []},
        }

        def fake_post(url, **kwargs):
            return _response(200, bodies[kwargs["files"]["audio"][0]])

        responses: list = []
        with patch.object(az.requests, "post", side_effect=fake_post):
            az.transcribe_chunks([(b, 60.0), (a, 0.0)], config=_config(), responses=responses)

        assert responses == [
            {"file": "chunk_000.ogg", "start_seconds": 0.0, "response": bodies["chunk_000.ogg"]},
            {"file": "chunk_001.ogg", "start_seconds": 60.0, "response": bodies["chunk_001.ogg"]},
        ]
        doc = az.response_document(_config(), responses)
        assert doc["api_version"] == az.DEFAULT_API_VERSION
        assert doc["definition"] == az.build_definition(_config())
        assert doc["chunks"] == responses
        assert "key" not in doc and "k" not in json.dumps(doc).split('"')

    def test_partial_responses_kept_on_failure(self, tmp_path: Path) -> None:
        a = tmp_path / "chunk_000.ogg"
        b = tmp_path / "chunk_001.ogg"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        ok = _response(200, {"phrases": [_phrase(0, 1000, "x")]})
        bad = _response(400, {"error": "bad audio"})
        responses: list = []
        with patch.object(az.requests, "post", side_effect=[ok, bad]):
            with pytest.raises(az.AzureTranscriptionError):
                az.transcribe_chunks([(a, 0.0), (b, 60.0)], config=_config(),
                                     responses=responses)
        assert [r["file"] for r in responses] == ["chunk_000.ogg"]
