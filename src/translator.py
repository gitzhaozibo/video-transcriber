"""GPT-based translation of transcription segments.

Uses the OpenAI API (``OPENAI_API_KEY`` environment variable) when available.
The API is intentionally optional: :func:`translate_segments` raises
:class:`TranslationUnavailableError` when no key is configured, and the app
catches that error so the workflow continues without translation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from src.subtitle import Segment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TARGET_LANGUAGE = "Japanese"


class TranslationUnavailableError(RuntimeError):
    """Raised when translation is requested but no OpenAI API key is set."""


class TranslationError(RuntimeError):
    """Raised when the OpenAI API call fails or returns unusable output."""


def _get_api_key() -> Optional[str]:
    """Return the OpenAI API key from env or Streamlit secrets."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:  # pragma: no cover - depends on streamlit runtime
        import streamlit as st

        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    return key


def is_available() -> bool:
    """Return True when an OpenAI API key is configured."""
    return bool(_get_api_key())


def translate_texts(
    texts: Sequence[str],
    *,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Translate *texts* into *target_language*, preserving order and count.

    All texts are sent in a single request separated by a ``\\n``-joined list
    so segment alignment is preserved.

    Raises
    ------
    TranslationUnavailableError
        If no API key is configured.
    TranslationError
        If the API call fails or the response cannot be aligned to the input.
    """
    key = _get_api_key()
    if not key:
        raise TranslationUnavailableError(
            "OPENAI_API_KEY is not set. Translation is skipped."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - defensive
        raise TranslationUnavailableError(
            "The 'openai' package is not installed. Translation is skipped."
        ) from exc

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"Translate the following numbered lines into {target_language}. "
        "Return ONLY the translated lines, keeping the same numbering and "
        "the same number of lines. Do not merge or split lines.\n\n"
        f"{numbered}"
    )

    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise TranslationError(f"OpenAI translation request failed: {exc}") from exc

    translations: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading "N." / "N)" / "N:" numbering if the model kept it.
        for sep in (". ", ") ", ": "):
            head, found, tail = line.partition(sep)
            if found and head.strip().isdigit():
                line = tail.strip()
                break
        translations.append(line)

    if len(translations) != len(texts):
        raise TranslationError(
            f"Translation returned {len(translations)} lines for "
            f"{len(texts)} inputs; cannot align segments."
        )
    return translations


def translate_segments(
    segments: Sequence[Segment],
    *,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    model: str = DEFAULT_MODEL,
    batch_size: int = 50,
) -> list[Segment]:
    """Translate the text of every segment, keeping timing untouched.

    Segments are translated in batches to avoid very large prompts.
    """
    if not segments:
        return []

    result: list[Segment] = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start : start + batch_size]
        translated = translate_texts(
            [seg.text for seg in batch],
            target_language=target_language,
            model=model,
        )
        for seg, text in zip(batch, translated):
            result.append(Segment(start=seg.start, end=seg.end, text=text))
    return result
