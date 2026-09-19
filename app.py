"""Video Transcriber – Streamlit application entry point.

Flow
----
1. Upload a video file (saved to the host-side job folder).
2. ``docker cp`` the video into the resident FFmpeg container.
3. Probe duration and extract audio inside the container (``exec_run``).
4. ``docker cp`` the extracted audio back to the Streamlit host.
5. Transcribe with faster-whisper (large-v3, CPU).
6. Translate the transcription with GPT (optional, needs OPENAI_API_KEY).
7. ``docker cp`` video + SRT to the container and burn subtitles (``exec_run``).
8. ``docker cp`` the subtitled video back and display it.
9. Transcription and translation text are both shown on screen.

Execution history for every step is written to both ``logs/app.jsonl`` and
PostgreSQL (``jobs`` / ``step_events`` tables), so it is possible to see
exactly which step a job is hanging on.
"""

from __future__ import annotations

import traceback
from pathlib import Path, PurePosixPath
from typing import Optional

import streamlit as st

from src import db
from src.ffmpeg_client import (
    CONTAINER_FONTS_DIR,
    HOST_SHARED_DIR,
    burn_subtitles,
    container_job_dir,
    copy_from_container,
    copy_to_container,
    extract_audio,
    get_docker_diagnostics,
    probe_duration,
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
from src.translator import (
    TranslationError,
    TranslationUnavailableError,
    is_available as translation_available,
    translate_segments,
)

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

_DEFAULTS: dict = {
    "job_id": None,
    "segments": None,
    "translated_segments": None,
    "video_duration": None,
    "original_filename": None,
    "output_bytes": None,
    "processing": False,
}
for _key, _default in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_session() -> None:
    """Clear all job-related session state."""
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default


def _check_fonts() -> bool:
    """Return True if at least one font file exists in FONTS_DIR."""
    if not FONTS_DIR.exists():
        return False
    for suffix in ("*.ttf", "*.otf", "*.ttc"):
        if any(FONTS_DIR.glob(suffix)):
            return True
    return False


def _fail(job_logger, step: str, exc: Exception, job_id: str, filename: str) -> None:
    """Log failure, mark the job failed, clean up, and reset the session."""
    st.error(f"{step} failed: {exc}")
    job_logger.failure(step, filename=filename, error=traceback.format_exc())
    db.update_job(job_id, status="failed", current_step=step, error=str(exc),
                  finished=True)
    cleanup_job(job_id, job_logger)
    _reset_session()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎬 Video Transcriber")
st.caption(
    "Upload → docker cp → FFmpeg container (probe & extract) → Whisper → "
    "GPT translation → docker cp → burn subtitles → docker cp → display"
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
        db.create_job(job_id, filename=uploaded_file.name)

        # Host-side paths inside shared/jobs/<uuid>/
        video_path = jdir / uploaded_file.name
        audio_path = jdir / (Path(uploaded_file.name).stem + ".m4a")

        # Container-side temp directory for this job
        cdir = container_job_dir(job_id)
        c_video = cdir / uploaded_file.name
        c_audio = cdir / (Path(uploaded_file.name).stem + ".m4a")

        failed_step: Optional[str] = None

        # ── Save upload ────────────────────────────────────────────────
        with st.spinner("Saving uploaded file…"):
            try:
                with StepTimer(job_logger, "upload", filename=uploaded_file.name):
                    video_path.write_bytes(uploaded_file.read())
            except Exception as exc:
                _fail(job_logger, "upload", exc, job_id, uploaded_file.name)
                failed_step = "upload"

        # ── docker cp: video → container ───────────────────────────────
        if failed_step is None:
            with st.spinner("Transferring video to FFmpeg container (docker cp)…"):
                try:
                    with StepTimer(
                        job_logger, "copy_to_container", filename=uploaded_file.name
                    ):
                        copy_to_container(video_path, cdir)
                except Exception as exc:
                    _fail(job_logger, "copy_to_container", exc, job_id,
                          uploaded_file.name)
                    failed_step = "copy_to_container"

        # ── Probe duration (in container) ──────────────────────────────
        if failed_step is None:
            with st.spinner("Probing video duration in container…"):
                try:
                    with StepTimer(
                        job_logger, "probe_duration", filename=uploaded_file.name
                    ):
                        duration = probe_duration(str(c_video))
                    st.session_state.video_duration = duration
                except Exception as exc:
                    _fail(job_logger, "probe_duration", exc, job_id,
                          uploaded_file.name)
                    failed_step = "probe_duration"

        # ── Extract audio (in container) ───────────────────────────────
        if failed_step is None:
            with st.spinner("Extracting audio in container (stream copy)…"):
                try:
                    with StepTimer(
                        job_logger,
                        "extract_audio",
                        filename=uploaded_file.name,
                        video_duration=st.session_state.video_duration,
                    ):
                        extract_audio(str(c_video), str(c_audio))
                except Exception as exc:
                    msg = str(exc)
                    if "audio" in msg.lower():
                        st.error(
                            "No audio track found in the uploaded video.  "
                            "Please upload a file with an audio stream."
                        )
                        job_logger.failure(
                            "extract_audio",
                            filename=uploaded_file.name,
                            error=msg,
                        )
                        db.update_job(job_id, status="failed",
                                      current_step="extract_audio",
                                      error=msg, finished=True)
                        cleanup_job(job_id, job_logger)
                        _reset_session()
                    else:
                        _fail(job_logger, "extract_audio", exc, job_id,
                              uploaded_file.name)
                    failed_step = "extract_audio"

        # ── docker cp: audio → host ────────────────────────────────────
        if failed_step is None:
            with st.spinner("Transferring audio back to server (docker cp)…"):
                try:
                    with StepTimer(
                        job_logger, "copy_from_container", filename=uploaded_file.name
                    ):
                        copy_from_container(PurePosixPath(c_audio), jdir)
                except Exception as exc:
                    _fail(job_logger, "copy_from_container", exc, job_id,
                          uploaded_file.name)
                    failed_step = "copy_from_container"

        # ── Transcribe ─────────────────────────────────────────────────
        if failed_step is None:
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
                            text=f"Transcribing… {progress*100:.0f}% "
                                 f"({seg.end:.1f}s / {duration:.1f}s)",
                        )
                progress_bar.progress(1.0, text="Transcription complete!")
                st.session_state.segments = segments_to_dicts(segments)
            except Exception as exc:
                _fail(job_logger, "transcribe", exc, job_id, uploaded_file.name)
                failed_step = "transcribe"

        # ── Translate (optional) ───────────────────────────────────────
        if failed_step is None:
            try:
                with st.spinner("Translating transcription with GPT…"):
                    with StepTimer(
                        job_logger, "translate", filename=uploaded_file.name
                    ):
                        translated = translate_segments(
                            segments_from_dicts(st.session_state.segments)
                        )
                st.session_state.translated_segments = segments_to_dicts(translated)
            except TranslationUnavailableError as exc:
                job_logger.info(
                    "translate",
                    filename=uploaded_file.name,
                    extra={"detail": f"skipped: {exc}"},
                )
                st.info(f"Translation skipped: {exc}")
            except TranslationError as exc:
                job_logger.failure(
                    "translate", filename=uploaded_file.name, error=str(exc)
                )
                st.warning(f"Translation failed (continuing without it): {exc}")

        if failed_step is None:
            db.update_job(job_id, status="transcribed", current_step="edit")
            st.session_state.processing = False
            st.rerun()

# ── Step 3: Review Transcription & Translation ──────────────────────────────

if st.session_state.segments is not None:
    st.header("③ Transcription & Translation")

    if st.session_state.video_duration:
        st.caption(
            f"Video duration: {st.session_state.video_duration:.1f} s  |  "
            f"Segments: {len(st.session_state.segments)}"
        )

    tab_transcript, tab_translation, tab_edit = st.tabs(
        ["📝 Transcription", "🌐 Translation", "✏ Edit (used for subtitles)"]
    )

    with tab_transcript:
        st.dataframe(
            st.session_state.segments,
            use_container_width=True,
            hide_index=True,
        )

    with tab_translation:
        if st.session_state.translated_segments is not None:
            st.dataframe(
                st.session_state.translated_segments,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No translation available.  Set the `OPENAI_API_KEY` "
                "environment variable to enable GPT translation."
            )

    with tab_edit:
        # The subtitles burned into the video default to the translation when
        # available, otherwise to the raw transcription.
        subtitle_source = (
            st.session_state.translated_segments
            if st.session_state.translated_segments is not None
            else st.session_state.segments
        )
        st.caption(
            "These segments become the burned-in subtitles "
            "(defaults to the translation when available)."
        )
        edited_rows = st.data_editor(
            subtitle_source,
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

    # ── Download SRTs ──────────────────────────────────────────────────
    with col_srt:
        stem = Path(st.session_state.original_filename or "output").stem
        srt_content = generate_srt(segments_from_dicts(edited_rows))
        st.download_button(
            label="⬇ Download Subtitle SRT",
            data=srt_content.encode("utf-8"),
            file_name=stem + ".srt",
            mime="text/plain",
        )
        transcript_srt = generate_srt(segments_from_dicts(st.session_state.segments))
        st.download_button(
            label="⬇ Download Transcription SRT",
            data=transcript_srt.encode("utf-8"),
            file_name=stem + "_transcript.srt",
            mime="text/plain",
        )

    # ── Burn subtitles ─────────────────────────────────────────────────
    with col_burn:
        st.subheader("④ Burn Subtitles into Video")

        has_fonts = _check_fonts()
        if not has_fonts:
            st.warning(
                "⚠ Japanese font not found in `shared/fonts/`.  "
                "CJK characters may not render correctly.  "
                "Place `NotoSansCJK-Regular.ttc` (or similar) in that folder."
            )

        if st.button(
            "🎬 Create Subtitled Video",
            disabled=st.session_state.processing
            or st.session_state.output_bytes is not None,
            type="primary",
        ):
            st.session_state.processing = True
            job_id = st.session_state.job_id
            jdir = job_dir(job_id)
            job_logger = get_logger(job_id)
            original_name = st.session_state.original_filename or "output.mp4"
            stem = Path(original_name).stem
            video_path = jdir / original_name
            srt_path = jdir / (stem + ".srt")
            output_path = jdir / (stem + "_subtitled.mp4")

            cdir = container_job_dir(job_id)
            c_video = cdir / original_name
            c_srt = cdir / (stem + ".srt")
            c_output = cdir / (stem + "_subtitled.mp4")

            burn_failed = False

            with st.spinner("Writing SRT file…"):
                write_srt(segments_from_dicts(edited_rows), srt_path)

            # ── docker cp: SRT → container (video is already there) ────
            with st.spinner("Transferring video & subtitle files to container (docker cp)…"):
                try:
                    with StepTimer(
                        job_logger, "copy_to_container", filename=srt_path.name
                    ):
                        copy_to_container(video_path, cdir)
                        copy_to_container(srt_path, cdir)
                except Exception as exc:
                    st.error(f"docker cp failed: {exc}")
                    job_logger.failure("copy_to_container", error=str(exc))
                    burn_failed = True

            # ── Burn subtitles (in container) ──────────────────────────
            if not burn_failed:
                with st.spinner("Burning subtitles (this may take a long time)…"):
                    try:
                        with StepTimer(
                            job_logger,
                            "burn_subtitles",
                            filename=original_name,
                            video_duration=st.session_state.video_duration,
                        ):
                            burn_subtitles(
                                str(c_video),
                                str(c_srt),
                                str(c_output),
                                fonts_dir=CONTAINER_FONTS_DIR if has_fonts else None,
                            )
                    except Exception as exc:
                        st.error(f"Subtitle burn failed: {exc}")
                        job_logger.failure("burn_subtitles", error=str(exc))
                        db.update_job(job_id, status="failed",
                                      current_step="burn_subtitles",
                                      error=str(exc), finished=True)
                        burn_failed = True

            # ── docker cp: output video → host ─────────────────────────
            if not burn_failed:
                with st.spinner("Transferring subtitled video back (docker cp)…"):
                    try:
                        with StepTimer(
                            job_logger, "copy_from_container", filename=c_output.name
                        ):
                            copy_from_container(PurePosixPath(c_output), jdir)
                        st.session_state.output_bytes = output_path.read_bytes()
                        db.update_job(job_id, status="completed", finished=True)
                    except Exception as exc:
                        st.error(f"docker cp failed: {exc}")
                        job_logger.failure("copy_from_container", error=str(exc))
                        db.update_job(job_id, status="failed",
                                      current_step="copy_from_container",
                                      error=str(exc), finished=True)

            st.session_state.processing = False
            st.rerun()

# ── Step 5: Display & download output ───────────────────────────────────────

if st.session_state.output_bytes is not None:
    st.success("✅ Subtitled video is ready!")
    st.video(st.session_state.output_bytes)
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
        "**FFmpeg container**: `ffmpeg-worker` (jrottenberg/ffmpeg)\n\n"
        "**Whisper model**: `large-v3` (CPU / int8)\n\n"
        f"**GPT translation**: {'enabled' if translation_available() else 'disabled (no OPENAI_API_KEY)'}\n\n"
        "**Font dir**: `shared/fonts/` → `/fonts` in container"
    )

    if st.button("🗑 Clean Up Current Job", disabled=st.session_state.job_id is None):
        job_id = st.session_state.job_id
        if job_id:
            cleanup_job(job_id, get_logger(job_id))
        _reset_session()
        st.rerun()

    # ── Execution history (PostgreSQL) ─────────────────────────────────
    st.divider()
    with st.expander("🗄 Execution History (DB)"):
        recent = db.list_recent_jobs(limit=10)
        if recent is None:
            st.info(
                "PostgreSQL is not available.  Start it with "
                "`docker compose up -d db`."
            )
        elif not recent:
            st.info("No jobs recorded yet.")
        else:
            st.dataframe(
                [
                    {
                        "job_id": r["job_id"][:8],
                        "file": r["filename"],
                        "status": r["status"],
                        "step": r["current_step"],
                        "updated": r["updated_at"],
                    }
                    for r in recent
                ],
                use_container_width=True,
                hide_index=True,
            )
            stuck = [r for r in recent if r["status"] == "running" and r["current_step"]]
            if stuck:
                st.warning(
                    "⚠ Jobs currently in-flight (check `current_step` for hangs): "
                    + ", ".join(f"{r['job_id'][:8]}@{r['current_step']}" for r in stuck)
                )

    # ── Docker diagnostics ─────────────────────────────────────────────
    with st.expander("🐳 Docker Diagnostics"):
        st.caption(
            "Collects docker version/info, container status, stats, processes "
            "and logs to help investigate a hung container or Docker Desktop "
            "crashes."
        )
        if st.button("Collect diagnostics"):
            with st.spinner("Collecting Docker diagnostics…"):
                sections = get_docker_diagnostics()
            for title, content in sections.items():
                st.markdown(f"**{title}**")
                st.code(content, language="text")
