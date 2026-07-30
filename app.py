"""Video Transcriber – Streamlit application entry point.

Flow
----
1. Upload a video file.
2. Extract audio via FFmpeg container and probe duration.
3. Transcribe with faster-whisper (large-v3, CPU).
4. Display and edit transcription segments.
5. Burn subtitles back into the video via FFmpeg container.
6. Download the output and clean up.
"""

from __future__ import annotations

import io
import traceback
from pathlib import Path
from typing import Optional

import streamlit as st

from src.ffmpeg_client import (
    HOST_SHARED_DIR,
    extract_audio,
    probe_duration,
    burn_subtitles,
)
from src.job_logger import StepTimer, get_logger
from src.jobs import cleanup_job, job_dir, new_job_id
from src.subtitle import (
    Segment,
    generate_srt,
    segments_from_dicts,
    segments_to_dicts,
    write_srt,
)
from src.transcriber import transcribe

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Video Transcriber",
    page_icon="🎬",
    layout="wide",
)

FONTS_DIR = HOST_SHARED_DIR / "fonts"

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "job_id" not in st.session_state:
    st.session_state.job_id: Optional[str] = None
if "segments" not in st.session_state:
    st.session_state.segments: Optional[list[dict]] = None
if "video_duration" not in st.session_state:
    st.session_state.video_duration: Optional[float] = None
if "original_filename" not in st.session_state:
    st.session_state.original_filename: Optional[str] = None
if "output_bytes" not in st.session_state:
    st.session_state.output_bytes: Optional[bytes] = None
if "processing" not in st.session_state:
    st.session_state.processing: bool = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_session() -> None:
    """Clear all job-related session state."""
    st.session_state.job_id = None
    st.session_state.segments = None
    st.session_state.video_duration = None
    st.session_state.original_filename = None
    st.session_state.output_bytes = None
    st.session_state.processing = False


def _check_fonts() -> bool:
    """Return True if at least one font file exists in FONTS_DIR."""
    if not FONTS_DIR.exists():
        return False
    for suffix in ("*.ttf", "*.otf", "*.ttc"):
        if any(FONTS_DIR.glob(suffix)):
            return True
    return False


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎬 Video Transcriber")
st.caption(
    "Upload a video → extract audio → transcribe with Whisper → edit → burn subtitles"
)

# ── Step 1: Upload ──────────────────────────────────────────────────────────

st.header("① Upload Video")

uploaded_file = st.file_uploader(
    "Select a video file (mp4, mov, avi, mkv, …)",
    type=["mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"],
    disabled=st.session_state.processing,
)

if uploaded_file is not None:
    # If a new file is uploaded while a previous job exists, reset.
    if st.session_state.original_filename != uploaded_file.name:
        _reset_session()

    st.session_state.original_filename = uploaded_file.name

    if st.button(
        "② Extract Audio & Transcribe",
        disabled=st.session_state.processing or st.session_state.segments is not None,
        type="primary",
    ):
        st.session_state.processing = True

        # ── Initialise job ─────────────────────────────────────────────
        job_id = new_job_id()
        st.session_state.job_id = job_id
        jdir = job_dir(job_id)
        job_logger = get_logger(job_id)

        # Paths inside shared/jobs/<uuid>/
        video_path = jdir / uploaded_file.name
        audio_path = jdir / (Path(uploaded_file.name).stem + ".m4a")
        srt_path = jdir / (Path(uploaded_file.name).stem + ".srt")

        error_occurred = False

        # ── Save upload ────────────────────────────────────────────────
        with st.spinner("Saving uploaded file…"):
            try:
                with StepTimer(job_logger, "upload", filename=uploaded_file.name):
                    video_path.write_bytes(uploaded_file.read())
            except Exception as exc:
                st.error(f"Failed to save upload: {exc}")
                job_logger.failure("upload", filename=uploaded_file.name, error=str(exc))
                cleanup_job(job_id)
                _reset_session()
                error_occurred = True

        if not error_occurred:
            # ── Probe duration ─────────────────────────────────────────
            with st.spinner("Probing video duration…"):
                try:
                    with StepTimer(
                        job_logger, "probe_duration", filename=uploaded_file.name
                    ):
                        duration = probe_duration(video_path)
                    st.session_state.video_duration = duration
                except RuntimeError as exc:
                    st.error(f"Could not read video duration: {exc}")
                    job_logger.failure(
                        "probe_duration", filename=uploaded_file.name, error=str(exc)
                    )
                    cleanup_job(job_id)
                    _reset_session()
                    error_occurred = True

        if not error_occurred:
            # ── Extract audio ──────────────────────────────────────────
            with st.spinner("Extracting audio (stream copy)…"):
                try:
                    with StepTimer(
                        job_logger,
                        "extract_audio",
                        filename=uploaded_file.name,
                        video_duration=st.session_state.video_duration,
                    ):
                        extract_audio(video_path, audio_path)
                except RuntimeError as exc:
                    msg = str(exc)
                    if "no audio" in msg.lower() or "audio" in msg.lower():
                        st.error(
                            "No audio track found in the uploaded video.  "
                            "Please upload a file with an audio stream."
                        )
                    else:
                        st.error(f"Audio extraction failed: {msg}")
                    cleanup_job(job_id)
                    _reset_session()
                    error_occurred = True

        if not error_occurred:
            # ── Transcribe ─────────────────────────────────────────────
            duration = st.session_state.video_duration or 1.0
            progress_bar = st.progress(0.0, text="Transcribing… (this may take a while)")
            segments: list[Segment] = []

            try:
                with StepTimer(
                    job_logger,
                    "transcribe",
                    filename=uploaded_file.name,
                    video_duration=duration,
                ):
                    for seg in transcribe(audio_path):
                        segments.append(seg)
                        progress = min(seg.end / duration, 1.0)
                        progress_bar.progress(
                            progress,
                            text=f"Transcribing… {progress*100:.0f}% ({seg.end:.1f}s / {duration:.1f}s)",
                        )
                progress_bar.progress(1.0, text="Transcription complete!")
                st.session_state.segments = segments_to_dicts(segments)
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                job_logger.failure(
                    "transcribe",
                    filename=uploaded_file.name,
                    error=traceback.format_exc(),
                )
                cleanup_job(job_id)
                _reset_session()
                error_occurred = True

        if not error_occurred:
            st.session_state.processing = False
            st.rerun()

# ── Step 4: Edit Segments ───────────────────────────────────────────────────

if st.session_state.segments is not None:
    st.header("③ Review & Edit Transcription")

    if st.session_state.video_duration:
        st.caption(
            f"Video duration: {st.session_state.video_duration:.1f} s  |  "
            f"Segments: {len(st.session_state.segments)}"
        )

    edited_rows = st.data_editor(
        st.session_state.segments,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "start": st.column_config.NumberColumn("Start (s)", format="%.3f"),
            "end": st.column_config.NumberColumn("End (s)", format="%.3f"),
            "text": st.column_config.TextColumn("Text", width="large"),
        },
        key="segment_editor",
    )

    col_srt, col_burn = st.columns(2)

    # ── Download SRT ───────────────────────────────────────────────────
    with col_srt:
        srt_content = generate_srt(segments_from_dicts(edited_rows))
        st.download_button(
            label="⬇ Download SRT",
            data=srt_content.encode("utf-8"),
            file_name=Path(st.session_state.original_filename or "output").stem + ".srt",
            mime="text/plain",
        )

    # ── Burn subtitles ─────────────────────────────────────────────────
    with col_burn:
        st.header("④ Burn Subtitles into Video")

        has_fonts = _check_fonts()
        if not has_fonts:
            st.warning(
                "⚠ Japanese font not found in `shared/fonts/`.  "
                "CJK characters may not render correctly.  "
                "Place `NotoSansCJK-Regular.ttc` (or similar) in that folder."
            )

        if st.button(
            "🎬 Create Subtitled Video",
            disabled=st.session_state.processing or st.session_state.output_bytes is not None,
            type="primary",
        ):
            st.session_state.processing = True
            job_id = st.session_state.job_id
            jdir = job_dir(job_id)
            job_logger = get_logger(job_id)
            original_name = st.session_state.original_filename or "output.mp4"
            video_path = jdir / original_name
            srt_path = jdir / (Path(original_name).stem + ".srt")
            output_path = jdir / (Path(original_name).stem + "_subtitled.mp4")

            with st.spinner("Writing SRT file…"):
                write_srt(segments_from_dicts(edited_rows), srt_path)

            with st.spinner("Burning subtitles (this may take a long time)…"):
                try:
                    with StepTimer(
                        job_logger,
                        "burn_subtitles",
                        filename=original_name,
                        video_duration=st.session_state.video_duration,
                    ):
                        burn_subtitles(
                            input_path=video_path,
                            srt_path=srt_path,
                            output_path=output_path,
                            fonts_dir=FONTS_DIR if has_fonts else None,
                        )
                    st.session_state.output_bytes = output_path.read_bytes()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    job_logger.failure("burn_subtitles", error=str(exc))
                except RuntimeError as exc:
                    st.error(f"Subtitle burn failed: {exc}")
                    job_logger.failure("burn_subtitles", error=str(exc))

            st.session_state.processing = False
            st.rerun()

# ── Step 5: Download output ─────────────────────────────────────────────────

if st.session_state.output_bytes is not None:
    st.success("✅ Subtitled video is ready!")
    stem = Path(st.session_state.original_filename or "output").stem
    st.download_button(
        label="⬇ Download Subtitled Video",
        data=st.session_state.output_bytes,
        file_name=f"{stem}_subtitled.mp4",
        mime="video/mp4",
        type="primary",
    )

    if st.button("🔄 Start New Job"):
        job_id = st.session_state.job_id
        if job_id:
            cleanup_job(job_id, get_logger(job_id))
        _reset_session()
        st.rerun()

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ Info")
    if st.session_state.job_id:
        st.caption(f"Job ID: `{st.session_state.job_id}`")
    if st.session_state.video_duration:
        m, s = divmod(int(st.session_state.video_duration), 60)
        h, m = divmod(m, 60)
        st.metric("Video Duration", f"{h:02d}:{m:02d}:{s:02d}")
    if st.session_state.segments is not None:
        st.metric("Segments", len(st.session_state.segments))

    st.divider()
    st.markdown(
        "**FFmpeg container**: `ffmpeg-worker`\n\n"
        "**Whisper model**: `large-v3` (CPU / int8)\n\n"
        "**Font dir**: `shared/fonts/`"
    )

    if st.button("🗑 Clean Up Current Job", disabled=st.session_state.job_id is None):
        job_id = st.session_state.job_id
        if job_id:
            cleanup_job(job_id, get_logger(job_id))
        _reset_session()
        st.rerun()
