"""Helpers for comparing transcription engines (used by scripts/compare_engines.py).

Everything here is pure (no Docker / network), so it is unit-testable: timing
bookkeeping, text normalisation and similarity, time-window alignment of the
two transcripts, and rendering of the console summary and HTML report.
"""

from __future__ import annotations

import difflib
import html
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.subtitle import Segment, format_timestamp

# Display names for timing stages, in report order.
STAGE_LABELS = {
    "extract": "音声抽出（コンテナ）",
    "copy": "コピー（コンテナ→ホスト）",
    "model_load": "モデル読み込み",
    "transcribe": "文字起こし",
}


@dataclass
class EngineRun:
    """Result of running one engine on the comparison input."""

    engine: str                      # "whisper" / "azure"
    detail: str = ""                 # e.g. model name or locales
    segments: list[Segment] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    error: Optional[str] = None
    raw_response: Optional[dict] = None  # Azure: response.json() with settings

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def speakers(self) -> set[int]:
        return {s.speaker for s in self.segments if s.speaker is not None}

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    def summary(self, duration: float, reference: Optional[str] = None) -> dict:
        """Return a JSON-serialisable summary of this run."""
        seg_lengths = [s.end - s.start for s in self.segments]
        result: dict = {
            "engine": self.engine,
            "detail": self.detail,
            "error": self.error,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "total_seconds": round(self.total_seconds, 3),
            "transcribe_rtf": _rtf(self.timings.get("transcribe"), duration),
            "total_rtf": _rtf(self.total_seconds if self.timings else None, duration),
            "segments": len(self.segments),
            "characters": len(normalize_text(self.text)),
            "avg_segment_seconds": (
                round(sum(seg_lengths) / len(seg_lengths), 2) if seg_lengths else None
            ),
            "max_segment_seconds": round(max(seg_lengths), 2) if seg_lengths else None,
            "speakers": len(self.speakers) if self.speakers else None,
        }
        if reference is not None and self.ok:
            result["reference_similarity"] = round(similarity(self.text, reference), 4)
        return result


def _rtf(seconds: Optional[float], duration: float) -> Optional[float]:
    if seconds is None or not duration:
        return None
    return round(seconds / duration, 4)


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lower-case, NFKC-normalise and drop whitespace/punctuation/symbols.

    Whisper and Azure differ in punctuation, spacing and full/half-width
    characters; removing those makes the similarity reflect the words.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] not in ("P", "Z", "S", "C")
    )


def similarity(a: str, b: str) -> float:
    """Character-level similarity (0–1) of two transcripts after normalisation."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Time-window alignment
# ---------------------------------------------------------------------------


def time_windows(
    runs: Sequence[EngineRun], window_seconds: float, duration: float
) -> list[tuple[float, float, dict[str, str]]]:
    """Group each engine's segments into fixed time windows (by segment start).

    Returns ``[(start, end, {engine: text}), ...]`` covering the whole
    *duration* so the transcripts can be read side by side.
    """
    window_seconds = max(float(window_seconds), 1.0)
    last_end = max(
        [duration] + [s.end for r in runs for s in r.segments], default=duration
    )
    count = max(int(last_end // window_seconds) + 1, 1)
    windows: list[tuple[float, float, dict[str, list[str]]]] = [
        (i * window_seconds, (i + 1) * window_seconds, {r.engine: [] for r in runs})
        for i in range(count)
    ]
    for run in runs:
        for seg in run.segments:
            idx = min(int(max(seg.start, 0.0) // window_seconds), count - 1)
            text = seg.text.strip()
            if text and seg.speaker is not None:
                text = f"[S{seg.speaker}] {text}"
            windows[idx][2][run.engine].append(text)
    return [
        (start, end, {engine: " ".join(t for t in texts if t) for engine, texts in by.items()})
        for start, end, by in windows
        if any(by.values())
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value >= 60:
        m, s = divmod(value, 60)
        h, m = divmod(int(m), 60)
        return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"
    return f"{value:.1f}s"


def _fmt_rtf(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def console_summary(
    runs: Sequence[EngineRun],
    duration: float,
    agreement: Optional[float],
    reference: Optional[str] = None,
) -> str:
    """Return a plain-text summary table for the terminal."""
    stages = [s for s in STAGE_LABELS if any(s in r.timings for r in runs)]
    header = ["", *[r.engine for r in runs]]
    rows: list[list[str]] = [["詳細", *[r.detail or "-" for r in runs]]]
    for stage in stages:
        rows.append([STAGE_LABELS[stage], *[_fmt_seconds(r.timings.get(stage)) for r in runs]])
    summaries = [r.summary(duration, reference) for r in runs]
    rows += [
        ["合計時間", *[_fmt_seconds(r.total_seconds) for r in runs]],
        ["RTF（文字起こし）", *[_fmt_rtf(s["transcribe_rtf"]) for s in summaries]],
        ["RTF（合計）", *[_fmt_rtf(s["total_rtf"]) for s in summaries]],
        ["セグメント数", *[str(s["segments"]) for s in summaries]],
        ["文字数（記号除く）", *[str(s["characters"]) for s in summaries]],
        ["平均セグメント長", *[_fmt_seconds(s["avg_segment_seconds"]) for s in summaries]],
        ["最大セグメント長", *[_fmt_seconds(s["max_segment_seconds"]) for s in summaries]],
        ["話者数", *[str(s["speakers"] or "-") for s in summaries]],
    ]
    if reference is not None:
        rows.append(
            ["正解との一致率", *[
                f"{s['reference_similarity']:.1%}" if "reference_similarity" in s else "-"
                for s in summaries
            ]]
        )
    rows.append(["エラー", *[r.error or "-" for r in runs]])

    def width(text: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)

    table = [header, *rows]
    widths = [max(width(row[i]) for row in table) for i in range(len(header))]
    lines = []
    for row in table:
        lines.append("  ".join(cell + " " * (widths[i] - width(cell)) for i, cell in enumerate(row)))
    lines.insert(1, "-" * len(lines[0]))
    lines.append("")
    lines.append(f"音声の長さ: {_fmt_seconds(duration)}")
    if agreement is not None:
        lines.append(f"エンジン間の一致率（参考値）: {agreement:.1%}")
    return "\n".join(lines)


def render_html(
    *,
    title: str,
    source_name: str,
    duration: float,
    runs: Sequence[EngineRun],
    agreement: Optional[float],
    window_seconds: float,
    reference: Optional[str] = None,
    notes: Sequence[str] = (),
) -> str:
    """Render a self-contained HTML comparison report."""
    esc = html.escape
    summaries = [r.summary(duration, reference) for r in runs]
    stages = [s for s in STAGE_LABELS if any(s in r.timings for r in runs)]

    def row(label: str, cells: Sequence[str], cls: str = "") -> str:
        tds = "".join(f'<td class="num">{c}</td>' for c in cells)
        return f'<tr class="{cls}"><th>{esc(label)}</th>{tds}</tr>'

    rows = [row("詳細", [esc(r.detail or "-") for r in runs])]
    rows += [
        row(STAGE_LABELS[s], [_fmt_seconds(r.timings.get(s)) for r in runs]) for s in stages
    ]
    rows += [
        row("合計時間", [_fmt_seconds(r.total_seconds) for r in runs], "strong"),
        row("RTF（文字起こし）", [_fmt_rtf(s["transcribe_rtf"]) for s in summaries], "strong"),
        row("RTF（合計）", [_fmt_rtf(s["total_rtf"]) for s in summaries]),
        row("セグメント数", [str(s["segments"]) for s in summaries]),
        row("文字数（記号除く）", [str(s["characters"]) for s in summaries]),
        row("平均セグメント長", [_fmt_seconds(s["avg_segment_seconds"]) for s in summaries]),
        row("最大セグメント長", [_fmt_seconds(s["max_segment_seconds"]) for s in summaries]),
        row("話者数", [str(s["speakers"] or "-") for s in summaries]),
    ]
    if reference is not None:
        rows.append(row("正解との一致率", [
            f"{s['reference_similarity']:.1%}" if "reference_similarity" in s else "-"
            for s in summaries
        ], "strong"))
    if any(r.error for r in runs):
        rows.append(row("エラー", [esc(r.error or "-") for r in runs], "error"))

    head_cells = "".join(f"<th>{esc(r.engine)}</th>" for r in runs)
    timeline_rows = []
    for start, end, texts in time_windows(runs, window_seconds, duration):
        cells = "".join(f"<td>{esc(texts.get(r.engine, ''))}</td>" for r in runs)
        stamp = f"{format_timestamp(start)[:8]}–{format_timestamp(min(end, max(duration, start)))[:8]}"
        timeline_rows.append(f'<tr><th class="time">{stamp}</th>{cells}</tr>')

    meta = [f"入力: {esc(source_name)}", f"音声の長さ: {_fmt_seconds(duration)}"]
    if agreement is not None:
        meta.append(f"エンジン間の一致率（参考値）: <b>{agreement:.1%}</b>")
    note_items = "".join(f"<li>{esc(n)}</li>" for n in notes)

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#1f2328; --muted:#656d76; --line:#d0d7de; --head:#f6f8fa; --err:#cf222e; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --head:#161b22; --err:#ff7b72; }}
}}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:14px/1.6 system-ui, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif; }}
main {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:28px 0 8px; }}
.meta {{ color:var(--muted); margin:0 0 8px; }}
.meta span + span::before {{ content:" · "; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ border:1px solid var(--line); padding:6px 10px; vertical-align:top; text-align:left; }}
thead th, tbody th {{ background:var(--head); font-weight:600; }}
td.num {{ font-variant-numeric:tabular-nums; }}
tr.strong td {{ font-weight:600; }}
tr.error td {{ color:var(--err); white-space:pre-wrap; }}
th.time {{ white-space:nowrap; font-variant-numeric:tabular-nums; font-weight:500; }}
.timeline td {{ width:45%; }}
ul {{ color:var(--muted); padding-left:20px; }}
</style>
</head>
<body><main>
<h1>{esc(title)}</h1>
<p class="meta">{"".join(f"<span>{m}</span>" for m in meta)}</p>
<h2>速度・統計</h2>
<div class="scroll"><table>
<thead><tr><th></th>{head_cells}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
<ul>
<li>RTF = 処理時間 ÷ 音声の長さ（1.0 未満なら実時間より速い）</li>
<li>一致率は記号・空白を除いた文字単位の類似度です（どちらが正しいかは示しません）</li>
{note_items}
</ul>
<h2>文字起こし結果（{int(window_seconds)} 秒ごと）</h2>
<div class="scroll"><table class="timeline">
<thead><tr><th>時間</th>{head_cells}</tr></thead>
<tbody>{"".join(timeline_rows)}</tbody>
</table></div>
</main></body>
</html>
"""
