"""Video Transcriber – Streamlit application entry point.

Flow
----
1. Upload a video file (saved to the host-side job folder).
2. ``docker cp`` the video into the resident FFmpeg container.
3. Probe duration and extract audio inside the container (``exec_run``).
4. ``docker cp`` the extracted audio back to the Streamlit host.
5. Transcribe with faster-whisper (large-v3, CPU), or with Azure AI Speech
   fast transcription when ``TRANSCRIBE_ENGINE=azure``.
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

from src.env import load_env

# Load .env before the other src modules read os.environ at import time.
load_env()

from src import db  # noqa: E402
from src.ffmpeg_client import (  # noqa: E402
    CONTAINER_FONTS_DIR,
    HOST_SHARED_DIR,
    SPEECH_AUDIO_FORMAT,
    burn_subtitles,
    container_job_dir,
    copy_from_container,
    copy_to_container,
    extract_audio,
    extract_speech_audio_chunks,
    get_docker_diagnostics,
    probe_duration,
)
from src import azure_transcriber, results  # noqa: E402
from src.job_logger import StepTimer, get_logger  # noqa: E402
from src.jobs import JOBS_ROOT, cleanup_job, job_dir, new_job_id  # noqa: E402
from src.subtitle import (  # noqa: E402
    Segment,
    generate_srt,
    segments_from_dicts,
    segments_to_dicts,
    write_srt,
)
from src.transcriber import (  # noqa: E402
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL as WHISPER_MODEL,
    ENGINE_AZURE,
    ENGINE_WHISPER,
    get_engine,
    transcribe,
)
from src.translator import (  # noqa: E402
    DEFAULT_MODEL as TRANSLATION_MODEL,
    DEFAULT_TARGET_LANGUAGE,
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
# Transcription engine
#
# The engine is chosen on screen; TRANSCRIBE_ENGINE (whisper|azure) only sets
# the initial selection.  Azure is selectable only when its settings are valid.
# ---------------------------------------------------------------------------

ENGINE_LABELS = {
    ENGINE_WHISPER: "Whisper (local, large-v3 / CPU)",
    ENGINE_AZURE: "Azure AI Speech (fast transcription)",
}

DEFAULT_ENGINE = ENGINE_WHISPER
ENGINE_ENV_ERROR: Optional[str] = None
try:
    DEFAULT_ENGINE = get_engine()
except ValueError as _exc:
    ENGINE_ENV_ERROR = str(_exc)

AZURE_CONFIG: Optional[azure_transcriber.AzureSpeechConfig] = None
AZURE_CONFIG_ERROR: Optional[str] = None
try:
    AZURE_CONFIG = azure_transcriber.load_config()
except azure_transcriber.AzureTranscriptionConfigError as _exc:
    AZURE_CONFIG_ERROR = str(_exc)

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
    "used_engine": None,  # engine that produced st.session_state.segments
    "result": None,       # per-job result document (src.results), saved as JSON
    "result_path": None,  # where that JSON was last saved
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


def _timer(job_logger, step: str, **kwargs) -> StepTimer:
    """StepTimer that also records the step in the current result document."""
    result = st.session_state.get("result")
    return StepTimer(
        job_logger, step, record=result["steps"] if result else None, **kwargs
    )


def _save_result() -> None:
    """Write the job's JSON files to RESULTS_DIR and the job folder (best effort)."""
    result = st.session_state.get("result")
    if result is None:
        return
    try:
        folder = results.save_result(result, job_folder=job_dir(result["job_id"]))
        st.session_state.result_path = str(folder)
    except OSError as exc:
        st.warning(f"Could not save the result JSON: {exc}")


def _fail_result(step: str, message: str) -> None:
    """Mark the current result document failed and save it."""
    result = st.session_state.get("result")
    if result is not None:
        results.set_error(result, step, message)
        _save_result()
        if st.session_state.result_path:
            st.caption(f"Result JSON saved: `{st.session_state.result_path}`")


def _engine_info(engine: str) -> dict:
    """Engine name and settings recorded in the result JSON (no secrets)."""
    if engine == ENGINE_AZURE and AZURE_CONFIG is not None:
        return {
            "name": ENGINE_AZURE,
            "api": "fast transcription",
            "api_version": AZURE_CONFIG.api_version,
            "endpoint": AZURE_CONFIG.endpoint,
            "locales": list(AZURE_CONFIG.locales),
            "diarization": AZURE_CONFIG.diarization,
            "max_speakers": AZURE_CONFIG.max_speakers if AZURE_CONFIG.diarization else None,
            "max_segment_seconds": AZURE_CONFIG.max_segment_seconds,
            "max_chunk_seconds": AZURE_CONFIG.max_chunk_seconds,
        }
    return {
        "name": ENGINE_WHISPER,
        "model": WHISPER_MODEL,
        "device": DEFAULT_DEVICE,
        "compute_type": DEFAULT_COMPUTE_TYPE,
    }


def _fail(job_logger, step: str, exc: Exception, job_id: str, filename: str) -> None:
    """Log failure, mark the job failed, clean up, and reset the session."""
    st.error(f"{step} failed: {exc}")
    job_logger.failure(step, filename=filename, error=traceback.format_exc())
    db.update_job(job_id, status="failed", current_step=step, error=str(exc),
                  finished=True)
    _fail_result(step, str(exc))
    cleanup_job(job_id, job_logger)
    _reset_session()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎬 Video Transcriber")
st.caption(
    "Upload → docker cp → FFmpeg container (probe & extract) → "
    "Whisper / Azure Speech → "
    "GPT translation → docker cp → burn subtitles → docker cp → display"
)

if ENGINE_ENV_ERROR:
    st.warning(f"{ENGINE_ENV_ERROR} — defaulting to Whisper.")

# ── Step 1: Upload ──────────────────────────────────────────────────────────

st.header("① Upload Video")

uploaded_file = st.file_uploader(
    "Select a video file (mp4, mov, avi, mkv, …)",
    type=["mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"],
    disabled=st.session_state.processing,
)

if "engine_choice" not in st.session_state:
    st.session_state.engine_choice = DEFAULT_ENGINE
selected_engine: str = st.radio(
    "Transcription engine",
    options=list(ENGINE_LABELS),
    format_func=ENGINE_LABELS.__getitem__,
    key="engine_choice",
    horizontal=True,
    disabled=st.session_state.processing or st.session_state.segments is not None,
    help="The initial selection comes from TRANSCRIBE_ENGINE in .env.",
)
engine_unavailable = selected_engine == ENGINE_AZURE and AZURE_CONFIG is None
if engine_unavailable:
    st.error(f"Azure AI Speech is not configured: {AZURE_CONFIG_ERROR}")
elif selected_engine == ENGINE_AZURE:
    st.caption(
        "Audio is sent to Azure (billed per audio hour).  "
        f"Language candidates: {', '.join(AZURE_CONFIG.locales)}  |  "
        + (
            f"Speaker identification: on (up to {AZURE_CONFIG.max_speakers})"
            if AZURE_CONFIG.diarization
            else "Speaker identification: off"
        )
    )
else:
    st.caption("Runs locally on CPU — roughly 1–3 h per hour of audio.")

if uploaded_file is not None:
    # If a new file is uploaded while a previous job exists, reset.
    if st.session_state.original_filename != uploaded_file.name:
        _reset_session()

    st.session_state.original_filename = uploaded_file.name

    if st.button(
        "② Extract Audio & Transcribe",
        disabled=st.session_state.processing
        or st.session_state.segments is not None
        or engine_unavailable,
        type="primary",
    ):
        st.session_state.processing = True
        use_azure = selected_engine == ENGINE_AZURE

        # ── Initialise job ─────────────────────────────────────────────
        job_id = new_job_id()
        st.session_state.job_id = job_id
        jdir = job_dir(job_id)
        job_logger = get_logger(job_id)
        db.create_job(job_id, filename=uploaded_file.name)
        job_logger.info(
            "transcribe",
            filename=uploaded_file.name,
            extra={"detail": f"engine={selected_engine}"},
        )
        st.session_state.result = results.new_result(
            job_id,
            filename=uploaded_file.name,
            size_bytes=getattr(uploaded_file, "size", None),
            engine=_engine_info(selected_engine),
        )
        result = st.session_state.result

        # Host-side paths inside shared/jobs/<uuid>/
        video_path = jdir / uploaded_file.name
        audio_path = jdir / (Path(uploaded_file.name).stem + ".m4a")

        # Container-side temp directory for this job
        cdir = container_job_dir(job_id)
        c_video = cdir / uploaded_file.name
        c_audio = cdir / (Path(uploaded_file.name).stem + ".m4a")
        # Azure: compact speech audio chunks (container → host)
        c_speech_dir = cdir / "speech"
        speech_dir = jdir / "speech"
        c_chunks: list = []
        host_chunks: list[tuple[Path, float]] = []

        failed_step: Optional[str] = None

        # ── Save upload ────────────────────────────────────────────────
        with st.spinner("Saving uploaded file…"):
            try:
                with _timer(job_logger, "upload", filename=uploaded_file.name):
                    video_path.write_bytes(uploaded_file.read())
            except Exception as exc:
                _fail(job_logger, "upload", exc, job_id, uploaded_file.name)
                failed_step = "upload"

        # ── docker cp: video → container ───────────────────────────────
        if failed_step is None:
            with st.spinner("Transferring video to FFmpeg container (docker cp)…"):
                try:
                    with _timer(
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
                    with _timer(
                        job_logger, "probe_duration", filename=uploaded_file.name
                    ):
                        duration = probe_duration(str(c_video))
                    st.session_state.video_duration = duration
                    result["input"]["duration_seconds"] = round(duration, 3)
                except Exception as exc:
                    _fail(job_logger, "probe_duration", exc, job_id,
                          uploaded_file.name)
                    failed_step = "probe_duration"

        # ── Extract audio (in container) ───────────────────────────────
        if failed_step is None:
            spinner_text = (
                "Extracting speech audio in container (mono 16 kHz Opus)…"
                if use_azure
                else "Extracting audio in container (stream copy)…"
            )
            with st.spinner(spinner_text):
                try:
                    with _timer(
                        job_logger,
                        "extract_audio",
                        filename=uploaded_file.name,
                        video_duration=st.session_state.video_duration,
                    ):
                        if use_azure:
                            # Straight from the video to mono 16 kHz Opus 32k.
                            c_chunks = extract_speech_audio_chunks(
                                str(c_video),
                                str(c_speech_dir),
                                chunk_seconds=AZURE_CONFIG.max_chunk_seconds,
                            )
                            result["engine"]["audio_format"] = dict(SPEECH_AUDIO_FORMAT)
                        else:
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
                        _fail_result("extract_audio", msg)
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
                    with _timer(
                        job_logger, "copy_from_container", filename=uploaded_file.name
                    ):
                        if use_azure:
                            host_chunks = [
                                (copy_from_container(chunk.path, speech_dir), chunk.start)
                                for chunk in c_chunks
                            ]
                            result["engine"]["audio_chunks"] = [
                                {
                                    "file": path.name,
                                    "start_seconds": round(start, 3),
                                    "size_bytes": path.stat().st_size,
                                }
                                for path, start in host_chunks
                            ]
                        else:
                            copy_from_container(PurePosixPath(c_audio), jdir)
                except Exception as exc:
                    _fail(job_logger, "copy_from_container", exc, job_id,
                          uploaded_file.name)
                    failed_step = "copy_from_container"

        # ── Transcribe (Azure fast transcription) ──────────────────────
        if failed_step is None and use_azure:
            spinner_text = (
                "Transcribing with Azure Speech (fast transcription)…"
                if len(host_chunks) == 1
                else f"Transcribing with Azure Speech ({len(host_chunks)} chunks)…"
            )
            try:
                with st.spinner(spinner_text):
                    with _timer(
                        job_logger,
                        "transcribe",
                        filename=uploaded_file.name,
                        video_duration=st.session_state.video_duration,
                    ):
                        raw_responses: list = []
                        try:
                            segments = azure_transcriber.transcribe_chunks(
                                host_chunks, config=AZURE_CONFIG,
                                responses=raw_responses,
                            )
                        finally:
                            # Keep response.json() of every chunk that came
                            # back, even if a later chunk failed.
                            if raw_responses:
                                results.set_raw_response(
                                    result,
                                    azure_transcriber.response_document(
                                        AZURE_CONFIG, raw_responses
                                    ),
                                )
                st.session_state.segments = segments_to_dicts(segments)
            except Exception as exc:
                _fail(job_logger, "transcribe", exc, job_id, uploaded_file.name)
                failed_step = "transcribe"

        # ── Transcribe (local Whisper) ─────────────────────────────────
        if failed_step is None and not use_azure:
            duration = st.session_state.video_duration or 1.0
            progress_bar = st.progress(0.0, text="Transcribing… (this may take a while)")
            segments: list[Segment] = []

            try:
                with _timer(
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
                    with _timer(
                        job_logger, "translate", filename=uploaded_file.name
                    ):
                        translated = translate_segments(
                            segments_from_dicts(st.session_state.segments)
                        )
                st.session_state.translated_segments = segments_to_dicts(translated)
                results.set_translation(
                    result, status="completed", model=TRANSLATION_MODEL,
                    target_language=DEFAULT_TARGET_LANGUAGE, segments=translated,
                )
            except TranslationUnavailableError as exc:
                job_logger.info(
                    "translate",
                    filename=uploaded_file.name,
                    extra={"detail": f"skipped: {exc}"},
                )
                results.set_translation(result, status="skipped", message=str(exc))
                st.info(f"Translation skipped: {exc}")
            except TranslationError as exc:
                job_logger.failure(
                    "translate", filename=uploaded_file.name, error=str(exc)
                )
                results.set_translation(
                    result, status="failed", model=TRANSLATION_MODEL,
                    target_language=DEFAULT_TARGET_LANGUAGE, message=str(exc),
                )
                st.warning(f"Translation failed (continuing without it): {exc}")

        if failed_step is None:
            db.update_job(job_id, status="transcribed", current_step="edit")
            st.session_state.used_engine = selected_engine
            results.set_transcription(result, st.session_state.segments)
            result["status"] = "transcribed"
            _save_result()
            st.session_state.processing = False
            st.rerun()

# ── Step 3: Review Transcription & Translation ──────────────────────────────

if st.session_state.segments is not None:
    st.header("③ Transcription & Translation")

    caption_parts = []
    if st.session_state.used_engine:
        caption_parts.append(f"Engine: {ENGINE_LABELS[st.session_state.used_engine]}")
    if st.session_state.video_duration:
        caption_parts.append(f"Video duration: {st.session_state.video_duration:.1f} s")
    caption_parts.append(f"Segments: {len(st.session_state.segments)}")
    st.caption("  |  ".join(caption_parts))

    tab_transcript, tab_translation, tab_edit = st.tabs(
        ["📝 Transcription", "🌐 Translation", "✏ Edit (used for subtitles)"]
    )

    # Show the speaker column only when the engine identified speakers (Azure
    # diarization); Whisper rows carry speaker=None.
    has_speakers = any(
        row.get("speaker") is not None for row in st.session_state.segments
    )
    segment_columns = (["speaker"] if has_speakers else []) + ["start", "end", "text"]
    segment_column_config = {
        "speaker": st.column_config.NumberColumn("Speaker", format="%d", step=1),
        "start": st.column_config.NumberColumn("Start (s)", format="%.3f"),
        "end": st.column_config.NumberColumn("End (s)", format="%.3f"),
        "text": st.column_config.TextColumn("Text", width="large"),
    }

    with tab_transcript:
        st.dataframe(
            st.session_state.segments,
            use_container_width=True,
            hide_index=True,
            column_order=segment_columns,
            column_config=segment_column_config,
        )

    with tab_translation:
        if st.session_state.translated_segments is not None:
            st.dataframe(
                st.session_state.translated_segments,
                use_container_width=True,
                hide_index=True,
                column_order=segment_columns,
                column_config=segment_column_config,
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
            column_order=segment_columns,
            column_config=segment_column_config,
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
        if st.session_state.result is not None:
            # Download reflects the current edits; the saved file is updated
            # when subtitles are burned.
            result_view = dict(
                results.result_document(st.session_state.result),
                subtitles={"segments": results.segments_payload(edited_rows)},
            )
            st.download_button(
                label="⬇ Download Result JSON",
                data=results.to_json(result_view).encode("utf-8"),
                file_name=results.result_filename(st.session_state.result),
                mime="application/json",
            )
        if st.session_state.result_path:
            st.caption(
                f"JSON saved to `{st.session_state.result_path}` and the job folder "
                f"`{JOBS_ROOT / st.session_state.job_id}` "
                "(result.json, transcript.json, …"
                + (", azure_response.json"
                   if results.RAW_RESPONSE_KEY in (st.session_state.result or {}) else "")
                + ")"
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
            if st.session_state.result is not None:
                st.session_state.result["subtitles"] = {
                    "segments": results.segments_payload(edited_rows)
                }

            # ── docker cp: SRT → container (video is already there) ────
            with st.spinner("Transferring video & subtitle files to container (docker cp)…"):
                try:
                    with _timer(
                        job_logger, "copy_to_container", filename=srt_path.name
                    ):
                        copy_to_container(video_path, cdir)
                        copy_to_container(srt_path, cdir)
                except Exception as exc:
                    st.error(f"docker cp failed: {exc}")
                    job_logger.failure("copy_to_container", error=str(exc))
                    _fail_result("copy_to_container", str(exc))
                    burn_failed = True

            # ── Burn subtitles (in container) ──────────────────────────
            if not burn_failed:
                with st.spinner("Burning subtitles (this may take a long time)…"):
                    try:
                        with _timer(
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
                        _fail_result("burn_subtitles", str(exc))
                        burn_failed = True

            # ── docker cp: output video → host ─────────────────────────
            if not burn_failed:
                with st.spinner("Transferring subtitled video back (docker cp)…"):
                    try:
                        with _timer(
                            job_logger, "copy_from_container", filename=c_output.name
                        ):
                            copy_from_container(PurePosixPath(c_output), jdir)
                        st.session_state.output_bytes = output_path.read_bytes()
                        db.update_job(job_id, status="completed", finished=True)
                        if st.session_state.result is not None:
                            st.session_state.result.update(
                                status="completed",
                                error=None,
                                output={"filename": output_path.name},
                            )
                            _save_result()
                    except Exception as exc:
                        st.error(f"docker cp failed: {exc}")
                        job_logger.failure("copy_from_container", error=str(exc))
                        db.update_job(job_id, status="failed",
                                      current_step="copy_from_container",
                                      error=str(exc), finished=True)
                        _fail_result("copy_from_container", str(exc))

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
        + f"**Transcription**: {ENGINE_LABELS[selected_engine]}\n\n"
        + (
            "**Azure Speech**: configured "
            f"({', '.join(AZURE_CONFIG.locales)})\n\n"
            if AZURE_CONFIG is not None
            else "**Azure Speech**: not configured\n\n"
        )
        + f"**GPT translation**: {'enabled' if translation_available() else 'disabled (no OPENAI_API_KEY)'}\n\n"
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
