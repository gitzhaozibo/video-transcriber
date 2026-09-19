"""FFmpeg / ffprobe client via Docker SDK against a resident container.

The ``ffmpeg-worker`` container (jrottenberg/ffmpeg) stays running all the
time.  Files are transferred with ``docker cp`` into a per-job temporary
directory inside the container (``/tmp/vt-jobs/<job_id>``), ffmpeg/ffprobe are
executed with ``container.exec_run``, and results are copied back with
``docker cp``.

A module-level :class:`threading.Lock` serialises container access so that
multiple concurrent Streamlit sessions do not overwhelm Docker Desktop /
Hyper-V with simultaneous exec/cp requests (a known cause of the container
becoming unresponsive).

The module also exposes :func:`get_docker_diagnostics` which collects
``docker info`` / ``docker stats`` / ``docker inspect`` output to help
investigate cases where the container stops responding or Docker Desktop
crashes.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTAINER_NAME = "ffmpeg-worker"
# Root of the shared volume on the host.  Only used for fonts and for
# host-side job folders; media files are docker-cp'd into the container.
HOST_SHARED_DIR = Path(os.environ.get("SHARED_DIR", "shared")).resolve()
CONTAINER_WORK_DIR = PurePosixPath("/work")
# Per-job temp directories inside the container.
CONTAINER_TEMP_ROOT = PurePosixPath("/tmp/vt-jobs")
# Fonts directory inside the container (read-only bind mount).
CONTAINER_FONTS_DIR = PurePosixPath("/fonts")

# Timeout (seconds) for a single exec_run call.  Long ffmpeg runs are allowed
# because subtitle burning re-encodes the whole video.
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", str(6 * 60 * 60)))

# Serialise Docker API access across Streamlit sessions/threads.
_docker_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def host_to_container(host_path: Path) -> str:
    """Convert a host-side path under *shared/* to a container-side path.

    Retained for backwards compatibility and for paths that still live on the
    bind-mounted volume (e.g. fonts).

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


def container_job_dir(job_id: str) -> PurePosixPath:
    """Return the container-side temp directory for *job_id* (not created)."""
    return CONTAINER_TEMP_ROOT / job_id


# ---------------------------------------------------------------------------
# Container client
# ---------------------------------------------------------------------------


def _get_client() -> docker.DockerClient:
    try:
        return docker.from_env()
    except docker.errors.DockerException as exc:
        raise RuntimeError(
            "Docker daemon is unreachable.  Make sure Docker Desktop is running."
        ) from exc


def _get_container() -> Container:
    """Return the running ``ffmpeg-worker`` container.

    Raises
    ------
    RuntimeError
        If the Docker daemon is unreachable or the container is not running.
    """
    try:
        client = _get_client()
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


def _exec(
    cmd: list[str],
    workdir: Optional[str] = None,
    timeout: Optional[int] = None,
) -> tuple[int, str]:
    """Run *cmd* inside the FFmpeg worker container (serialised by lock).

    Returns (exit_code, combined stdout+stderr).
    """
    timeout = timeout if timeout is not None else EXEC_TIMEOUT
    logger.debug("exec in %s: %s (timeout=%ss)", CONTAINER_NAME, cmd, timeout)
    started = time.monotonic()
    with _docker_lock:
        container = _get_container()
        try:
            exec_id = container.client.api.exec_create(
                container.id,
                cmd=cmd,
                workdir=workdir,
            )["Id"]
            output = container.client.api.exec_start(exec_id, demux=False, socket=False)
            deadline = started + timeout
            while True:
                info = container.client.api.exec_inspect(exec_id)
                if not info.get("Running"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Command timed out after {timeout}s inside container "
                        f"'{CONTAINER_NAME}': {cmd!r}"
                    )
                time.sleep(0.2)
            exit_code = info.get("ExitCode", 1) or 0
        except docker.errors.APIError as exc:
            raise RuntimeError(
                f"Docker API error while executing {cmd!r} in "
                f"'{CONTAINER_NAME}': {exc}"
            ) from exc
    text = output.decode("utf-8", errors="replace") if output else ""
    logger.debug(
        "exec finished in %.1fs (exit=%d): %s",
        time.monotonic() - started,
        exit_code,
        cmd,
    )
    return exit_code, text


# ---------------------------------------------------------------------------
# docker cp equivalents
# ---------------------------------------------------------------------------


def _make_tar(file_path: Path, arcname: str) -> bytes:
    """Return *file_path* packaged as a tar archive named *arcname*."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(str(file_path), arcname=arcname)
    return buf.getvalue()


def copy_to_container(
    host_path: Path, container_dest_dir: PurePosixPath
) -> PurePosixPath:
    """Copy a host file into the container (equivalent of ``docker cp``).

    Parameters
    ----------
    host_path:
        Existing file on the host.
    container_dest_dir:
        Directory inside the container to copy the file into (created).

    Returns
    -------
    PurePosixPath
        Full path of the copied file inside the container.
    """
    host_path = Path(host_path)
    if not host_path.exists():
        raise FileNotFoundError(f"Host file not found: {host_path}")

    with _docker_lock:
        container = _get_container()
        try:
            container.exec_run(["mkdir", "-p", str(container_dest_dir)])
            archive = _make_tar(host_path, host_path.name)
            ok = container.put_archive(str(container_dest_dir), archive)
        except docker.errors.APIError as exc:
            raise RuntimeError(
                f"docker cp into '{CONTAINER_NAME}' failed for {host_path}: {exc}"
            ) from exc
    if not ok:
        raise RuntimeError(
            f"docker cp into '{CONTAINER_NAME}' returned failure for {host_path}"
        )
    dest = container_dest_dir / host_path.name
    logger.debug("copied %s -> %s:%s", host_path, CONTAINER_NAME, dest)
    return dest


def copy_from_container(container_path: PurePosixPath, host_dest_dir: Path) -> Path:
    """Copy a file from the container to the host (equivalent of ``docker cp``).

    Returns the host-side path of the copied file.
    """
    container_path = PurePosixPath(container_path)
    host_dest_dir = Path(host_dest_dir)
    host_dest_dir.mkdir(parents=True, exist_ok=True)
    dest = host_dest_dir / container_path.name

    with _docker_lock:
        container = _get_container()
        try:
            stream, _stat = container.get_archive(str(container_path))
        except docker.errors.NotFound as exc:
            raise FileNotFoundError(
                f"File not found in container '{CONTAINER_NAME}': {container_path}"
            ) from exc
        except docker.errors.APIError as exc:
            raise RuntimeError(
                f"docker cp from '{CONTAINER_NAME}' failed for {container_path}: {exc}"
            ) from exc

        raw = b"".join(stream)

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
        member = tar.getmembers()[0]
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract {container_path} from archive")
        dest.write_bytes(extracted.read())
    logger.debug("copied %s:%s -> %s", CONTAINER_NAME, container_path, dest)
    return dest


def remove_container_path(container_path: PurePosixPath) -> None:
    """Remove a file or directory inside the container (``rm -rf``)."""
    exit_code, output = _exec(["rm", "-rf", str(container_path)])
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to remove {container_path} in '{CONTAINER_NAME}': {output}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_duration(container_input: str) -> float:
    """Return the duration of a media file (container-side path) in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        container_input,
    ]
    exit_code, output = _exec(cmd)
    if exit_code != 0:
        raise RuntimeError(f"ffprobe failed (exit={exit_code}):\n{output}")
    stripped = output.strip()
    try:
        return float(stripped)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse ffprobe duration output: {stripped!r}"
        ) from exc


def extract_audio(container_input: str, container_output: str) -> None:
    """Extract the audio stream (stream copy, no re-encode) in the container."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", container_input,
        "-vn",
        "-acodec", "copy",
        container_output,
    ]
    exit_code, output = _exec(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit={exit_code}):\n{output}"
        )


def burn_subtitles(
    container_input: str,
    container_srt: str,
    container_output: str,
    *,
    fonts_dir: Optional[PurePosixPath] = CONTAINER_FONTS_DIR,
    font_name: str = "Noto Sans CJK JP",
) -> None:
    """Burn *container_srt* subtitles into *container_input* in the container."""
    subtitles_filter = f"subtitles={container_srt}"
    if fonts_dir is not None:
        subtitles_filter += (
            f":fontsdir={fonts_dir}"
            f":force_style='FontName={font_name}'"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", container_input,
        "-vf", subtitles_filter,
        "-c:a", "copy",
        container_output,
    ]
    exit_code, output = _exec(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"ffmpeg subtitle burn failed (exit={exit_code}):\n{output}"
        )


# ---------------------------------------------------------------------------
# Diagnostics (for investigating hung containers / Docker Desktop crashes)
# ---------------------------------------------------------------------------


def get_docker_diagnostics() -> dict[str, str]:
    """Collect diagnostics useful when the container or Docker Desktop hangs.

    Returns a mapping of section title -> command output.  Every section is
    best-effort; failures are recorded as the section's text.
    """
    sections: dict[str, str] = {}

    def _capture(title: str, fn: Callable[[], str]) -> None:
        try:
            sections[title] = fn()
        except Exception as exc:
            sections[title] = f"<failed to collect: {exc}>"

    _capture("docker version", lambda: str(_get_client().version()))
    _capture("docker info", lambda: str(_get_client().info()))

    def _container_status() -> str:
        c = _get_client().containers.get(CONTAINER_NAME)
        c.reload()
        return (
            f"status={c.status} image={c.image.tags} "
            f"started_at={c.attrs['State'].get('StartedAt')}"
        )

    _capture(f"container '{CONTAINER_NAME}' status", _container_status)

    def _stats() -> str:
        c = _get_client().containers.get(CONTAINER_NAME)
        stats = c.stats(stream=False)
        cpu = stats.get("cpu_stats", {})
        mem = stats.get("memory_stats", {})
        usage = mem.get("usage", 0)
        limit = mem.get("limit", 0)
        return (
            f"cpu_total_usage={cpu.get('cpu_usage', {}).get('total_usage')}\n"
            f"memory_usage={usage / 2**20:.1f} MiB / {limit / 2**30:.1f} GiB"
        )

    _capture(f"container '{CONTAINER_NAME}' stats", _stats)

    def _procs() -> str:
        c = _get_client().containers.get(CONTAINER_NAME)
        top = c.top()
        rows = top.get("Processes") or []
        return "\n".join(" ".join(map(str, p)) for p in rows) or "<no processes>"

    _capture(f"container '{CONTAINER_NAME}' processes", _procs)
    _capture(
        f"container '{CONTAINER_NAME}' logs (tail 100)",
        lambda: _get_client()
        .containers.get(CONTAINER_NAME)
        .logs(tail=100)
        .decode("utf-8", errors="replace"),
    )
    return sections
