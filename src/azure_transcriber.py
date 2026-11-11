"""Azure AI Speech *fast transcription* backend (REST, whole-file upload).

Selected with ``TRANSCRIBE_ENGINE=azure``.  Each audio file is POSTed to the
fast transcription API in a single synchronous request and the returned
``phrases`` are converted to the same :class:`~src.subtitle.Segment` objects
the faster-whisper backend produces, so the rest of the pipeline (SRT, GPT
translation, editing, burn-in) is unchanged.

Service limits (Standard S0): < 500 MB and < 5 hours per file.  The app
therefore sends compact mono 16 kHz Opus audio (about 14 MB per hour) that
the FFmpeg container splits into chunks of ``AZURE_MAX_CHUNK_SECONDS``; the
chunk offsets are added back here so timestamps refer to the whole video.

Configuration (environment variables or Streamlit secrets):

``AZURE_SPEECH_KEY``            Speech resource key (required).
``AZURE_SPEECH_ENDPOINT``       e.g. ``https://<name>.cognitiveservices.azure.com``.
``AZURE_SPEECH_REGION``         e.g. ``japaneast`` (used when no endpoint is set).
``AZURE_SPEECH_LOCALES``        Candidate locales for language identification, or
                                ``multilingual`` for code-switched audio (see below).
``AZURE_PHRASE_LIST``           Words to recognise better (names, products...),
                                comma or newline separated.
``AZURE_PHRASE_LIST_WEIGHT``    Optional phrase list biasing weight (0.0–2.0).
``AZURE_DIARIZATION``           Identify speakers (``true``/``false``, default true).
``AZURE_MAX_SPEAKERS``          Upper bound hint for diarization (2–35, default 4).
``AZURE_MAX_SEGMENT_SECONDS``   Split phrases longer than this (0 disables).
``AZURE_MAX_CHUNK_SECONDS``     Audio chunk length sent per request.
``AZURE_REQUEST_TIMEOUT``       Read timeout (seconds) for one request.

Language identification picks **one main locale per request** (per audio
chunk here).  For audio that switches language mid-file, set
``AZURE_SPEECH_LOCALES=multilingual``: ``locales`` is then sent empty and the
multilingual model transcribes each language as it is spoken (it supports
ja-JP, en-US, zh-CN and de-DE among others).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from src.subtitle import Segment, parse_speaker

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2025-10-15"
DEFAULT_LOCALES = ("ja-JP", "en-US", "zh-CN", "de-DE")
DEFAULT_MAX_SEGMENT_SECONDS = 10.0
DEFAULT_DIARIZATION = True
DEFAULT_MAX_SPEAKERS = 4
MULTILINGUAL = "multilingual"   # AZURE_SPEECH_LOCALES value for locales: []
PHRASE_WEIGHT_RANGE = (0.0, 2.0)
MIN_SPEAKERS, MAX_SPEAKERS = 2, 35  # range accepted by the API
# Fast transcription accepts < 5 h per file; 4 h chunks leave a safe margin.
API_MAX_AUDIO_SECONDS = 5 * 60 * 60
DEFAULT_MAX_CHUNK_SECONDS = 4 * 60 * 60
DEFAULT_REQUEST_TIMEOUT = 60 * 60
CONNECT_TIMEOUT = 30
MAX_ATTEMPTS = 3
_RETRY_STATUS = {429, 500, 502, 503, 504}

_MIME_TYPES = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}

# Patched in tests to avoid real waiting between retries.
_sleep = time.sleep


class AzureTranscriptionConfigError(RuntimeError):
    """Raised when the Azure Speech settings are missing or invalid."""


class AzureTranscriptionError(RuntimeError):
    """Raised when the fast transcription request fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _get_setting(name: str) -> Optional[str]:
    """Return *name* from the environment, falling back to Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:  # pragma: no cover - depends on streamlit runtime
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else None


def _number_setting(name: str, default: float) -> float:
    raw = _get_setting(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise AzureTranscriptionConfigError(f"{name} must be a number: {raw!r}") from exc


def _bool_setting(name: str, default: bool) -> bool:
    raw = _get_setting(name)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise AzureTranscriptionConfigError(f"{name} must be true or false: {raw!r}")


@dataclass(frozen=True)
class AzureSpeechConfig:
    key: str
    endpoint: str
    api_version: str = DEFAULT_API_VERSION
    locales: tuple[str, ...] = DEFAULT_LOCALES
    diarization: bool = DEFAULT_DIARIZATION
    max_speakers: int = DEFAULT_MAX_SPEAKERS
    max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS
    max_chunk_seconds: int = DEFAULT_MAX_CHUNK_SECONDS
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    phrase_list: tuple[str, ...] = ()
    phrase_list_weight: Optional[float] = None

    @property
    def multilingual(self) -> bool:
        """True when ``locales`` is sent empty (multilingual transcription)."""
        return not self.locales

    @property
    def language_mode(self) -> str:
        """``multilingual``, ``identification`` (several candidates) or ``fixed``."""
        if self.multilingual:
            return MULTILINGUAL
        return "identification" if len(self.locales) > 1 else "fixed"

    @property
    def language_label(self) -> str:
        """Human readable language setting, e.g. for the UI."""
        if self.multilingual:
            return "multilingual (language may switch within the audio)"
        return ", ".join(self.locales)

    @property
    def url(self) -> str:
        return (
            f"{self.endpoint}/speechtotext/transcriptions:transcribe"
            f"?api-version={self.api_version}"
        )


def load_config() -> AzureSpeechConfig:
    """Build the configuration from environment variables / secrets.

    Raises
    ------
    AzureTranscriptionConfigError
        If the key or endpoint/region is missing, or a value is invalid.
    """
    key = _get_setting("AZURE_SPEECH_KEY")
    if not key:
        raise AzureTranscriptionConfigError(
            "AZURE_SPEECH_KEY is not set (required when TRANSCRIBE_ENGINE=azure)."
        )

    endpoint = _get_setting("AZURE_SPEECH_ENDPOINT")
    region = _get_setting("AZURE_SPEECH_REGION")
    if endpoint:
        endpoint = endpoint.rstrip("/")
    elif region:
        endpoint = f"https://{region}.api.cognitive.microsoft.com"
    else:
        raise AzureTranscriptionConfigError(
            "Set AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION "
            "(required when TRANSCRIBE_ENGINE=azure)."
        )

    locales_raw = _get_setting("AZURE_SPEECH_LOCALES")
    if locales_raw and locales_raw.lower() == MULTILINGUAL:
        locales: tuple[str, ...] = ()
    elif locales_raw:
        locales = tuple(loc.strip() for loc in locales_raw.split(",") if loc.strip())
    else:
        locales = DEFAULT_LOCALES
    if MULTILINGUAL in (loc.lower() for loc in locales):
        raise AzureTranscriptionConfigError(
            f"AZURE_SPEECH_LOCALES={MULTILINGUAL} cannot be combined with locales: "
            f"{locales_raw!r}"
        )

    phrases_raw = _get_setting("AZURE_PHRASE_LIST") or ""
    phrase_list = tuple(
        dict.fromkeys(p.strip() for p in re.split(r"[,\n]", phrases_raw) if p.strip())
    )
    weight_raw = _get_setting("AZURE_PHRASE_LIST_WEIGHT")
    phrase_weight = None
    if weight_raw is not None:
        phrase_weight = _number_setting("AZURE_PHRASE_LIST_WEIGHT", 0.0)
        low, high = PHRASE_WEIGHT_RANGE
        if not low <= phrase_weight <= high:
            raise AzureTranscriptionConfigError(
                f"AZURE_PHRASE_LIST_WEIGHT must be between {low} and {high}: {phrase_weight}"
            )

    max_segment = _number_setting("AZURE_MAX_SEGMENT_SECONDS", DEFAULT_MAX_SEGMENT_SECONDS)
    max_chunk = int(_number_setting("AZURE_MAX_CHUNK_SECONDS", DEFAULT_MAX_CHUNK_SECONDS))
    if not 0 < max_chunk < API_MAX_AUDIO_SECONDS:
        raise AzureTranscriptionConfigError(
            f"AZURE_MAX_CHUNK_SECONDS must be between 1 and {API_MAX_AUDIO_SECONDS - 1} "
            f"(fast transcription accepts < 5 hours per file): {max_chunk}"
        )
    timeout = int(_number_setting("AZURE_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT))
    diarization = _bool_setting("AZURE_DIARIZATION", DEFAULT_DIARIZATION)
    max_speakers = int(_number_setting("AZURE_MAX_SPEAKERS", DEFAULT_MAX_SPEAKERS))
    if not MIN_SPEAKERS <= max_speakers <= MAX_SPEAKERS:
        raise AzureTranscriptionConfigError(
            f"AZURE_MAX_SPEAKERS must be between {MIN_SPEAKERS} and {MAX_SPEAKERS}: "
            f"{max_speakers}"
        )

    return AzureSpeechConfig(
        key=key,
        endpoint=endpoint,
        api_version=_get_setting("AZURE_SPEECH_API_VERSION") or DEFAULT_API_VERSION,
        locales=locales,
        diarization=diarization,
        max_speakers=max_speakers,
        max_segment_seconds=max_segment,
        max_chunk_seconds=max_chunk,
        request_timeout=timeout,
        phrase_list=phrase_list,
        phrase_list_weight=phrase_weight,
    )


# ---------------------------------------------------------------------------
# Response → Segment conversion
# ---------------------------------------------------------------------------


def _ms(value: Any) -> float:
    return float(value or 0) / 1000.0


def _split_phrase(
    phrase: dict, offset: float, max_segment_seconds: float
) -> list[Segment]:
    """Convert one API phrase to one or more segments.

    Phrases longer than *max_segment_seconds* are split at word boundaries
    using the word-level timestamps.  The split text is sliced from the
    phrase's display text so punctuation is preserved; if the words cannot be
    aligned to the text, the phrase is kept whole.
    """
    text = (phrase.get("text") or "").strip()
    if not text:
        return []

    speaker = parse_speaker(phrase.get("speaker"))
    start = offset + _ms(phrase.get("offsetMilliseconds"))
    end = start + _ms(phrase.get("durationMilliseconds"))
    whole = [Segment(start=start, end=end, text=text, speaker=speaker)]

    words = phrase.get("words") or []
    if max_segment_seconds <= 0 or end - start <= max_segment_seconds or len(words) < 2:
        return whole

    # Locate each word in the display text (in order).
    positions: list[int] = []
    cursor = 0
    for word in words:
        word_text = (word.get("text") or "").strip()
        idx = text.find(word_text, cursor) if word_text else -1
        if idx < 0:
            return whole
        positions.append(idx)
        cursor = idx + len(word_text)

    def word_start(i: int) -> float:
        return offset + _ms(words[i].get("offsetMilliseconds"))

    def word_end(i: int) -> float:
        return word_start(i) + _ms(words[i].get("durationMilliseconds"))

    # Group consecutive words so each group spans <= max_segment_seconds.
    groups: list[tuple[int, int]] = []
    first = 0
    for i in range(1, len(words)):
        if word_end(i) - word_start(first) > max_segment_seconds:
            groups.append((first, i - 1))
            first = i
    groups.append((first, len(words) - 1))

    segments: list[Segment] = []
    for gi, (a, b) in enumerate(groups):
        is_first = gi == 0
        is_last = gi == len(groups) - 1
        char_start = 0 if is_first else positions[a]
        char_end = len(text) if is_last else positions[groups[gi + 1][0]]
        piece = text[char_start:char_end].strip()
        if not piece:
            continue
        segments.append(
            Segment(
                start=start if is_first else word_start(a),
                end=end if is_last else word_end(b),
                text=piece,
                speaker=speaker,
            )
        )
    return segments or whole


def phrases_to_segments(
    result: dict,
    *,
    offset: float = 0.0,
    max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
) -> list[Segment]:
    """Convert a fast transcription response body to :class:`Segment` objects.

    *offset* (seconds) is added to every timestamp, which is how chunked audio
    is mapped back onto the timeline of the original video.
    """
    phrases = sorted(
        result.get("phrases") or [],
        key=lambda p: float(p.get("offsetMilliseconds") or 0),
    )
    segments: list[Segment] = []
    for phrase in phrases:
        segments.extend(_split_phrase(phrase, offset, max_segment_seconds))
    return segments


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return 5.0 * (2 ** (attempt - 1))


def build_definition(config: AzureSpeechConfig) -> dict[str, Any]:
    """Return the ``definition`` JSON sent with each fast transcription request."""
    definition: dict[str, Any] = {
        # Empty list = multilingual transcription; several = language identification.
        "locales": list(config.locales),
        "profanityFilterMode": "None",
    }
    if config.diarization:
        # The audio is mono, which diarization requires.
        definition["diarization"] = {"enabled": True, "maxSpeakers": config.max_speakers}
    if config.phrase_list:
        phrase_list: dict[str, Any] = {"phrases": list(config.phrase_list)}
        if config.phrase_list_weight is not None:
            phrase_list["biasingWeight"] = config.phrase_list_weight
        definition["phraseList"] = phrase_list
    return definition


def _post_audio(audio_path: Path, config: AzureSpeechConfig) -> dict:
    """POST *audio_path* to the fast transcription API and return the JSON body."""
    definition = json.dumps(build_definition(config))
    mime = _MIME_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with audio_path.open("rb") as audio_file:
                response = requests.post(
                    config.url,
                    headers={"Ocp-Apim-Subscription-Key": config.key},
                    files={
                        "audio": (audio_path.name, audio_file, mime),
                        "definition": (None, definition),
                    },
                    timeout=(CONNECT_TIMEOUT, config.request_timeout),
                )
        except requests.ConnectionError as exc:
            if attempt < MAX_ATTEMPTS:
                delay = _retry_delay(None, attempt)
                logger.warning("Azure connection error (%s); retrying in %.0fs", exc, delay)
                _sleep(delay)
                continue
            raise AzureTranscriptionError(
                f"Could not connect to Azure Speech ({config.endpoint}): {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise AzureTranscriptionError(
                f"Azure fast transcription request failed: {exc}"
            ) from exc

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise AzureTranscriptionError(
                    "Azure fast transcription returned a non-JSON response"
                ) from exc

        if response.status_code in _RETRY_STATUS and attempt < MAX_ATTEMPTS:
            delay = _retry_delay(response, attempt)
            logger.warning(
                "Azure fast transcription HTTP %d; retrying in %.0fs",
                response.status_code,
                delay,
            )
            _sleep(delay)
            continue

        raise AzureTranscriptionError(
            f"Azure fast transcription failed (HTTP {response.status_code}): "
            f"{response.text[:500]}"
        )

    raise AzureTranscriptionError("Azure fast transcription failed")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_file(
    audio_path: Path,
    *,
    offset: float = 0.0,
    config: Optional[AzureSpeechConfig] = None,
    responses: Optional[list] = None,
) -> list[Segment]:
    """Transcribe one audio file and return its segments.

    When *responses* is given, the raw response body (``response.json()``)
    is appended to it as ``{"file", "start_seconds", "response"}``.

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist.
    AzureTranscriptionConfigError
        If the configuration is missing or invalid.
    AzureTranscriptionError
        If the API request fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    config = config or load_config()

    logger.info(
        "Azure fast transcription of %s (%.1f MB, offset=%.1fs, locales=%s)",
        audio_path,
        audio_path.stat().st_size / 1_000_000,
        offset,
        config.language_label,
    )
    result = _post_audio(audio_path, config)
    if responses is not None:
        responses.append(
            {"file": audio_path.name, "start_seconds": round(offset, 3), "response": result}
        )
    return phrases_to_segments(
        result, offset=offset, max_segment_seconds=config.max_segment_seconds
    )


def transcribe_chunks(
    chunks: Sequence[tuple[Path, float]],
    *,
    config: Optional[AzureSpeechConfig] = None,
    responses: Optional[list] = None,
) -> list[Segment]:
    """Transcribe ``(audio_path, start_offset_seconds)`` chunks in order.

    Returns all segments on the original (un-chunked) timeline.  *responses*
    collects each chunk's raw response body (see :func:`transcribe_file`);
    on failure it keeps the responses of the chunks that succeeded.
    """
    config = config or load_config()
    segments: list[Segment] = []
    for audio_path, offset in sorted(chunks, key=lambda c: c[1]):
        segments.extend(
            transcribe_file(audio_path, offset=offset, config=config, responses=responses)
        )
    return segments


def locale_summary(responses: Sequence[dict]) -> dict[str, Any]:
    """Summarise the locales Azure detected, from raw chunk responses.

    Returns ``{"locale": <main locale or None>, "locales": [{"locale",
    "seconds", "phrases"}, ...]}`` with the locales ordered by speaking time.
    """
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for chunk in responses:
        for phrase in (chunk.get("response") or {}).get("phrases") or []:
            if not (phrase.get("text") or "").strip():
                continue
            locale = phrase.get("locale") or "unknown"
            seconds[locale] = seconds.get(locale, 0.0) + _ms(phrase.get("durationMilliseconds"))
            counts[locale] = counts.get(locale, 0) + 1
    ordered = sorted(seconds, key=lambda loc: (-seconds[loc], loc))
    return {
        "locale": ordered[0] if ordered else None,
        "locales": [
            {"locale": loc, "seconds": round(seconds[loc], 3), "phrases": counts[loc]}
            for loc in ordered
        ],
    }


def response_document(config: AzureSpeechConfig, responses: Sequence[dict]) -> dict[str, Any]:
    """Wrap raw chunk responses with the request settings (for saving as JSON).

    ``offsetMilliseconds`` inside each ``response`` is relative to its chunk;
    add the chunk's ``start_seconds`` to place it on the video's timeline.
    """
    return {
        "api": "fast transcription",
        "api_version": config.api_version,
        "endpoint": config.endpoint,
        "definition": build_definition(config),
        "chunks": list(responses),
    }
