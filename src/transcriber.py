"""faster-whisper transcription wrapper.

Provides :func:`transcribe` which yields :class:`~src.subtitle.Segment`
objects incrementally so that the caller can update a progress bar in real
time.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Optional

from src.subtitle import Segment

logger = logging.getLogger(__name__)

# Default model configuration (CPU + int8 quantisation for speed)
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"

# Module-level cache so the large model is only loaded once per process.
_model_cache: dict[str, object] = {}


def _get_model(model_size: str, device: str, compute_type: str) -> object:
    """Load (or return cached) a WhisperModel.

    Parameters
    ----------
    model_size:
        Model identifier, e.g. ``"large-v3"``.
    device:
        ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation type, e.g. ``"int8"``.
    """
    cache_key = f"{model_size}:{device}:{compute_type}"
    if cache_key not in _model_cache:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        logger.info(
            "Loading faster-whisper model %s (device=%s, compute_type=%s)",
            model_size,
            device,
            compute_type,
        )
        _model_cache[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _model_cache[cache_key]


def transcribe(
    audio_path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: Optional[str] = None,
    beam_size: int = 5,
) -> Generator[Segment, None, None]:
    """Transcribe *audio_path* and yield :class:`~src.subtitle.Segment` objects.

    Segments are yielded incrementally as faster-whisper processes them, which
    lets the caller update a progress indicator.

    Parameters
    ----------
    audio_path:
        Path to the audio file (e.g. ``.m4a`` / ``.mp3``).
    model_size:
        faster-whisper model identifier (default: ``"large-v3"``).
    device:
        ``"cpu"`` or ``"cuda"`` (default: ``"cpu"``).
    compute_type:
        Quantisation (default: ``"int8"`` for CPU).
    language:
        BCP-47 language code (e.g. ``"ja"``), or ``None`` for auto-detect.
    beam_size:
        Beam search width (default: 5).

    Yields
    ------
    Segment
        One segment per caption unit with start/end times and text.

    Raises
    ------
    RuntimeError
        If faster-whisper raises an unexpected error.
    FileNotFoundError
        If *audio_path* does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model(model_size, device, compute_type)

    logger.info("Starting transcription of %s (language=%s)", audio_path, language)

    try:
        segments_iter, _info = model.transcribe(  # type: ignore[attr-defined]
            str(audio_path),
            beam_size=beam_size,
            language=language,
        )
    except Exception as exc:
        raise RuntimeError(f"faster-whisper transcription failed: {exc}") from exc

    for raw_seg in segments_iter:
        yield Segment(
            start=raw_seg.start,
            end=raw_seg.end,
            text=raw_seg.text,
        )
