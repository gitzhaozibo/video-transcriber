"""SRT subtitle file generation and time-formatting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Segment:
    """A single transcription segment."""

    start: float  # seconds
    end: float    # seconds
    text: str


def format_timestamp(seconds: float) -> str:
    """Convert *seconds* to SRT timestamp format ``HH:MM:SS,mmm``.

    Parameters
    ----------
    seconds:
        Non-negative number of seconds (fractional part becomes milliseconds).

    Returns
    -------
    str
        Timestamp string, e.g. ``"00:01:23,456"``.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(segments: Sequence[Segment]) -> str:
    """Generate an SRT file content string from *segments*.

    Parameters
    ----------
    segments:
        Ordered sequence of :class:`Segment` objects.

    Returns
    -------
    str
        Complete SRT file content (UTF-8 text).
    """
    lines: list[str] = []
    for index, seg in enumerate(segments, start=1):
        start_ts = format_timestamp(seg.start)
        end_ts = format_timestamp(seg.end)
        text = seg.text.strip()
        lines.append(f"{index}\n{start_ts} --> {end_ts}\n{text}\n")
    return "\n".join(lines)


def write_srt(segments: Sequence[Segment], path: "str | __import__('pathlib').Path") -> None:
    """Write an SRT file to *path*.

    Parameters
    ----------
    segments:
        Ordered sequence of :class:`Segment` objects.
    path:
        Destination file path.
    """
    from pathlib import Path

    content = generate_srt(segments)
    Path(path).write_text(content, encoding="utf-8")


def segments_from_dicts(rows: list[dict]) -> list[Segment]:
    """Convert a list of dicts (as produced by ``st.data_editor``) to segments.

    Each dict must contain ``"start"``, ``"end"``, and ``"text"`` keys.

    Parameters
    ----------
    rows:
        List of row dicts.

    Returns
    -------
    list[Segment]
        Parsed segments.
    """
    result: list[Segment] = []
    for row in rows:
        result.append(
            Segment(
                start=float(row["start"]),
                end=float(row["end"]),
                text=str(row["text"]),
            )
        )
    return result


def segments_to_dicts(segments: Sequence[Segment]) -> list[dict]:
    """Convert segments to a list of dicts suitable for ``st.data_editor``.

    Parameters
    ----------
    segments:
        Ordered sequence of :class:`Segment` objects.

    Returns
    -------
    list[dict]
        List of row dicts with keys ``"start"``, ``"end"``, and ``"text"``.
    """
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]
