"""FFmpeg / ffprobe client via Docker SDK.

Executes ffmpeg and ffprobe inside the running ``ffmpeg-worker`` container
(linuxserver/ffmpeg) using ``container.exec_run``.  All file paths are
expected in container-space (``/work/…``); callers use the path-conversion
helpers in this module to translate Windows/host paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath
from typing import Optional

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTAINER_NAME = "ffmpeg-worker"
HOST_SHARED_DIR = Path(os.environ.get("SHARED_DIR", "shared")).resolve()
CONTAINER_WORK_DIR = PurePosixPath("/work")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def host_to_container(host_path: Path) -> str:
    """Convert a host-side path under *shared/* to a container-side path.

    Parameters
    ----------
    host_path:
        Absolute path on the host that must be inside ``HOST_SHARED_DIR``.

    Returns
    -------
    str
        The corresponding ``/work/…`` path as a POSIX string.

    Raises
    ------
    ValueError
        If *host_path* is not inside ``HOST_SHARED_DIR``.
    """
    host_path = Path(host_path).resolve()
    try:
        relative = host_path.relative_to(HOST_SHARED_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Path {host_path!r} is not inside shared directory {HOST_SHARED_DIR!r}"
        ) from exc
    return str(CONTAINER_WORK_DIR / relative.as_posix())


def container_to_host(container_path: str) -> Path:
    """Convert a container-side ``/work/…`` path to a host-side path.

    Parameters
    ----------
    container_path:
        POSIX path inside the container that starts with ``/work``.

    Returns
    -------
    Path
        The corresponding host path inside ``HOST_SHARED_DIR``.

    Raises
    ------
    ValueError
        If *container_path* is not under ``/work``.
    """
    posix = PurePosixPath(container_path)
    try:
        relative = posix.relative_to(CONTAINER_WORK_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Container path {container_path!r} is not under {CONTAINER_WORK_DIR!r}"
        ) from exc
    return HOST_SHARED_DIR / str(relative)


# ---------------------------------------------------------------------------
# Container client
# ---------------------------------------------------------------------------


def _get_container() -> Container:
    """Return the running ``ffmpeg-worker`` container.

    Raises
    ------
    RuntimeError
        If the Docker daemon is unreachable or the container is not running.
    """
    try:
        client = docker.from_env()
        container = client.containers.get(CONTAINER_NAME)
    except docker.errors.DockerException as exc:
        raise RuntimeError(
            "Docker daemon is unreachable.  Make sure Docker Desktop is running."
        ) from exc
    except docker.errors.NotFound:
        raise RuntimeError(
            f"Container '{CONTAINER_NAME}' not found.  Run: docker compose up -d"
        )

    if container.status != "running":
        raise RuntimeError(
            f"Container '{CONTAINER_NAME}' is not running (status={container.status!r})."
            "  Run: docker compose up -d"
        )
    return container


def _exec(cmd: list[str], workdir: str = "/work") -> tuple[int, str]:
    """Run *cmd* inside the FFmpeg worker container.

    Parameters
    ----------
    cmd:
        Command and arguments to execute.
    workdir:
        Working directory inside the container.

    Returns
    -------
    (exit_code, output):
        Combined stdout+stderr as a single string.
    """
    container = _get_container()
    exit_code, output = container.exec_run(
        cmd=cmd,
        workdir=workdir,
        demux=False,
    )
    text = output.decode("utf-8", errors="replace") if output else ""
    return exit_code, text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_audio(
    input_path: Path,
    output_path: Path,
) -> None:
    """Extract the audio stream from *input_path* into *output_path*.

    Uses stream-copy (``-acodec copy``) so no re-encoding takes place.
    The output container is inferred from *output_path*'s suffix (typically
    ``.m4a`` for AAC streams).

    Parameters
    ----------
    input_path:
        Host path to the source video file (inside ``shared/``).
    output_path:
        Host path for the output audio file (inside ``shared/``).

    Raises
    ------
    RuntimeError
        On container errors or non-zero ffmpeg exit codes.
    ValueError
        If paths are outside the shared directory.
    """
    src = host_to_container(input_path)
    dst = host_to_container(output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", src,
        "-vn",
        "-acodec", "copy",
        dst,
    ]
    logger.debug("extract_audio cmd: %s", cmd)
    exit_code, output = _exec(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit={exit_code}):\n{output}"
        )


def probe_duration(input_path: Path) -> float:
    """Return the duration of *input_path* in seconds via ffprobe.

    Parameters
    ----------
    input_path:
        Host path to the video/audio file (inside ``shared/``).

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    RuntimeError
        On container errors, non-zero exit codes, or unparseable output.
    ValueError
        If path is outside the shared directory.
    """
    src = host_to_container(input_path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        src,
    ]
    logger.debug("probe_duration cmd: %s", cmd)
    exit_code, output = _exec(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"ffprobe failed (exit={exit_code}):\n{output}"
        )
    stripped = output.strip()
    try:
        return float(stripped)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse ffprobe duration output: {stripped!r}"
        ) from exc


def burn_subtitles(
    input_path: Path,
    srt_path: Path,
    output_path: Path,
    fonts_dir: Optional[Path] = None,
    font_name: str = "Noto Sans CJK JP",
) -> None:
    """Burn *srt_path* subtitles into *input_path*, writing to *output_path*.

    Parameters
    ----------
    input_path:
        Host path to the source video (inside ``shared/``).
    srt_path:
        Host path to the ``.srt`` subtitle file (inside ``shared/``).
    output_path:
        Host path for the output video (inside ``shared/``).
    fonts_dir:
        Host path to the directory containing CJK fonts.  When provided,
        passed as ``fontsdir=…`` to the ``subtitles`` filter.
    font_name:
        Font name for the ``force_style`` option.

    Raises
    ------
    RuntimeError
        On container errors or non-zero ffmpeg exit codes.
    FileNotFoundError
        If *fonts_dir* is provided but does not exist on the host.
    ValueError
        If paths are outside the shared directory.
    """
    if fonts_dir is not None and not fonts_dir.exists():
        raise FileNotFoundError(
            f"Fonts directory not found: {fonts_dir}\n"
            "Please place 'NotoSansCJK-Regular.ttc' (or similar) into "
            f"{fonts_dir} and restart."
        )

    src = host_to_container(input_path)
    srt = host_to_container(srt_path)
    dst = host_to_container(output_path)

    # Build the subtitles filter string
    # Escape colons in paths for the filtergraph (Windows paths may contain
    # drive letters, but inside the container paths are POSIX so this is safe).
    subtitles_filter = f"subtitles={srt}"
    if fonts_dir is not None:
        fonts_container = host_to_container(fonts_dir)
        subtitles_filter += (
            f":fontsdir={fonts_container}"
            f":force_style='FontName={font_name}'"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", src,
        "-vf", subtitles_filter,
        "-c:a", "copy",
        dst,
    ]
    logger.debug("burn_subtitles cmd: %s", cmd)
    exit_code, output = _exec(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"ffmpeg subtitle burn failed (exit={exit_code}):\n{output}"
        )
