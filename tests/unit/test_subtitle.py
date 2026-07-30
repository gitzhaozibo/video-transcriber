"""Unit tests for src.subtitle (SRT generation and time formatting)."""

from __future__ import annotations

import pytest

from src.subtitle import (
    Segment,
    format_timestamp,
    generate_srt,
    segments_from_dicts,
    segments_to_dicts,
    write_srt,
)


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_sub_second(self) -> None:
        assert format_timestamp(0.5) == "00:00:00,500"

    def test_seconds(self) -> None:
        assert format_timestamp(65.0) == "00:01:05,000"

    def test_hours(self) -> None:
        assert format_timestamp(3661.0) == "01:01:01,000"

    def test_milliseconds(self) -> None:
        assert format_timestamp(1.234) == "00:00:01,234"

    def test_negative_clamps_to_zero(self) -> None:
        assert format_timestamp(-5.0) == "00:00:00,000"

    def test_large_value(self) -> None:
        # 3 hours, 59 minutes, 59 seconds, 999 ms
        seconds = 3 * 3600 + 59 * 60 + 59 + 0.999
        result = format_timestamp(seconds)
        assert result == "03:59:59,999"


# ---------------------------------------------------------------------------
# generate_srt
# ---------------------------------------------------------------------------


class TestGenerateSrt:
    def test_empty(self) -> None:
        result = generate_srt([])
        assert result == ""

    def test_single_segment(self) -> None:
        segs = [Segment(start=0.0, end=1.5, text="Hello")]
        result = generate_srt(segs)
        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:01,500\n" in result
        assert "Hello" in result

    def test_multiple_segments_indexed(self) -> None:
        segs = [
            Segment(start=0.0, end=1.0, text="First"),
            Segment(start=1.5, end=3.0, text="Second"),
        ]
        result = generate_srt(segs)
        lines = result.splitlines()
        assert lines[0] == "1"
        # Find second block: search for "2\n" entry
        assert "2\n00:00:01,500 --> 00:00:03,000\nSecond" in result

    def test_text_is_stripped(self) -> None:
        segs = [Segment(start=0.0, end=1.0, text="  hello  ")]
        result = generate_srt(segs)
        assert "hello" in result
        assert "  hello  " not in result

    def test_japanese_text(self) -> None:
        segs = [Segment(start=0.0, end=2.0, text="これはテストです")]
        result = generate_srt(segs)
        assert "これはテストです" in result


# ---------------------------------------------------------------------------
# segments_from_dicts / segments_to_dicts
# ---------------------------------------------------------------------------


class TestSegmentConversions:
    def test_roundtrip(self) -> None:
        segs = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.5, end=3.0, text="World"),
        ]
        dicts = segments_to_dicts(segs)
        assert dicts == [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "World"},
        ]
        restored = segments_from_dicts(dicts)
        assert len(restored) == 2
        assert restored[0].start == 0.0
        assert restored[1].text == "World"

    def test_from_dicts_coerces_types(self) -> None:
        rows = [{"start": "0.5", "end": "1.5", "text": 42}]
        segs = segments_from_dicts(rows)
        assert segs[0].start == pytest.approx(0.5)
        assert segs[0].end == pytest.approx(1.5)
        assert segs[0].text == "42"


# ---------------------------------------------------------------------------
# write_srt
# ---------------------------------------------------------------------------


class TestWriteSrt:
    def test_writes_file(self, tmp_path: "pathlib.Path") -> None:
        from pathlib import Path

        segs = [Segment(start=0.0, end=1.0, text="Test")]
        dest = tmp_path / "output.srt"
        write_srt(segs, dest)
        content = dest.read_text(encoding="utf-8")
        assert "Test" in content
        assert "00:00:00,000 --> 00:00:01,000" in content

    def test_utf8_encoding(self, tmp_path: "pathlib.Path") -> None:
        from pathlib import Path

        segs = [Segment(start=0.0, end=1.0, text="日本語テスト")]
        dest = tmp_path / "output.srt"
        write_srt(segs, dest)
        content = dest.read_text(encoding="utf-8")
        assert "日本語テスト" in content
