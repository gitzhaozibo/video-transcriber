"""Azure AI Speech *fast transcription* backend (REST, whole-file upload).

Selected with ``TRANSCRIBE_ENGINE=azure``.  The audio file is POSTed to the
fast transcription API in a single synchronous request (no splitting) and the
returned ``phrases`` are converted to the same :class:`~src.subtitle.Segment`
objects the faster-whisper backend produces, so the rest of the pipeline (SRT,
GPT translation, editing, burn-in) is unchanged.

Service limits (Standard S0): < 500 MB and < 5 hours per file.  The app sends
compact mono 16 kHz Opus audio (about 15 MB per hour), so the 5-hour limit is
the one that matters; longer videos are rejected before upload.

The language is identified automatically among :data:`LOCALES`
(ja-JP, en-US, zh-CN, de-DE).

Configuration (environment variables or Streamlit secrets):

``AZURE_SPEECH_KEY``            Speech resource key (required).
``AZURE_SPEECH_ENDPOINT``       e.g. ``https://<name>.cognitiveservices.azure.com``.
``AZURE_SPEECH_REGION``         e.g. ``japaneast`` (used when no endpoint is set).
``AZURE_PHRASE_LIST``           Words to recognise better (names, products...),
                                comma or newline separated.
``AZURE_DIARIZATION``           Identify speakers (``true``/``false``, default true).
``AZURE_MAX_SPEAKERS``          Upper bound hint for diarization (2-35, default 4).
                                The API has no minimum-speakers setting.
``AZURE_REQUEST_TIMEOUT``       Read timeout (seconds) for the request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from src.subtitle import Segment, parse_speaker

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2025-10-15"
# Candidate locales for automatic language identification (fixed).
LOCALES = ("ja-JP", "en-US", "zh-CN", "de-DE")
DEFAULT_DIARIZATION = True
DEFAULT_MAX_SPEAKERS = 4
MIN_SPEAKERS, MAX_SPEAKERS = 2, 35  # range accepted by the API
# Fast transcription accepts < 5 h and < 500 MB per file.
API_MAX_AUDIO_SECONDS = 5 * 60 * 60
API_MAX_FILE_BYTES = 500 * 1000 * 1000
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
    locales: tuple[str, ...] = LOCALES
    diarization: bool = DEFAULT_DIARIZATION
    max_speakers: int = DEFAULT_MAX_SPEAKERS
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    phrase_list: tuple[str, ...] = ()

    @property
    def language_label(self) -> str:
        """Human readable language setting, e.g. for the UI."""
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

    phrases_raw = _get_setting("AZURE_PHRASE_LIST") or ""
    phrase_list = tuple(
        dict.fromkeys(p.strip() for p in re.split(r"[,\n]", phrases_raw) if p.strip())
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
        diarization=diarization,
        max_speakers=max_speakers,
        request_timeout=timeout,
        phrase_list=phrase_list,
    )


# ---------------------------------------------------------------------------
# Response → Segment conversion
# ---------------------------------------------------------------------------


def _ms(value: Any) -> float:
    return float(value or 0) / 1000.0


def phrase_to_segment(phrase: dict) -> Optional[Segment]:
    """Convert one API phrase to a :class:`Segment` (``None`` for empty text).

    Each phrase becomes exactly one segment, as returned by Azure.
    """
    text = (phrase.get("text") or "").strip()
    if not text:
        return None
    start = _ms(phrase.get("offsetMilliseconds"))
    return Segment(
        start=start,
        end=start + _ms(phrase.get("durationMilliseconds")),
        text=text,
        speaker=parse_speaker(phrase.get("speaker")),
    )


def phrases_to_segments(result: dict) -> list[Segment]:
    """Convert a fast transcription response body to :class:`Segment` objects."""
    phrases = sorted(
        result.get("phrases") or [],
        key=lambda p: float(p.get("offsetMilliseconds") or 0),
    )
    segments = (phrase_to_segment(phrase) for phrase in phrases)
    return [seg for seg in segments if seg is not None]


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
    """Return the ``definition`` JSON sent with the fast transcription request."""
    definition: dict[str, Any] = {
        # Several candidates = automatic language identification.
        "locales": list(config.locales),
        "profanityFilterMode": "None",
    }
    if config.diarization:
        # The audio is mono, which diarization requires.  The API takes only
        # an upper bound (maxSpeakers); there is no minimum.
        definition["diarization"] = {"enabled": True, "maxSpeakers": config.max_speakers}
    if config.phrase_list:
        definition["phraseList"] = {"phrases": list(config.phrase_list)}
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
    config: Optional[AzureSpeechConfig] = None,
) -> tuple[list[Segment], dict]:
    """Transcribe one audio file in a single request.

    Returns ``(segments, response)`` where *response* is the raw response
    body (``response.json()``).

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist.
    AzureTranscriptionConfigError
        If the configuration is missing or invalid.
    AzureTranscriptionError
        If the file is too large for the API or the request fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    config = config or load_config()

    size = audio_path.stat().st_size
    if size >= API_MAX_FILE_BYTES:
        raise AzureTranscriptionError(
            f"Audio file is too large for fast transcription "
            f"({size / 1_000_000:.0f} MB; the limit is < 500 MB per file)."
        )

    logger.info(
        "Azure fast transcription of %s (%.1f MB, locales=%s)",
        audio_path,
        size / 1_000_000,
        config.language_label,
    )
    response = _post_audio(audio_path, config)
    segments = phrases_to_segments(response)
    return segments, response


def locale_summary(response: dict) -> dict[str, Any]:
    """Summarise the locales Azure detected, from the raw response body.

    Returns ``{"locale": <main locale or None>, "locales": [{"locale",
    "seconds", "phrases"}, ...]}`` with the locales ordered by speaking time.
    """
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for phrase in (response or {}).get("phrases") or []:
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


def response_document(
    config: AzureSpeechConfig, audio_file: str, response: dict
) -> dict[str, Any]:
    """Wrap the raw response body with the request settings (saved as JSON)."""
    return {
        "api": "fast transcription",
        "api_version": config.api_version,
        "endpoint": config.endpoint,
        "definition": build_definition(config),
        "file": audio_file,
        "response": response,
    }
