"""Integration tests for FFmpeg container operations (docker cp workflow).

These tests require a running ``ffmpeg-worker`` Docker container (started
with ``docker compose up -d``).  They are marked with
``pytest.mark.integration`` and skipped automatically when the container is
not available.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import pytest

# ---------------------------------------------------------------------------
# Skip marker
# ---------------------------------------------------------------------------


def _container_available() -> bool:
    """Return True if the ffmpeg-worker container is running."""
    try:
        import docker

        client = docker.from_env()
        container = client.containers.get("ffmpeg-worker")
        return container.status == "running"
    except Exception:
        return False


pytestmark = pytest.mark.integration

skip_if_no_container = pytest.mark.skipif(
    not _container_available(),
    reason="ffmpeg-worker container is not running",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Host-side directory holding the generated test video and outputs."""
    return tmp_path_factory.mktemp("host-jobs")


@pytest.fixture(scope="module")
def test_video(work_dir: Path) -> Path:
    """Generate a 5-second test video using the host ffmpeg binary."""
    video_path = work_dir / "test_video.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=320x240:rate=25",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=5",
            "-shortest",
            str(video_path),
        ],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0 and not video_path.exists():
        pytest.skip("Could not generate test video: ffmpeg not available on host")
    return video_path


@pytest.fixture()
def container_dir() -> PurePosixPath:
    """A unique container-side temp directory, cleaned up after the test."""
    import uuid

    from src.ffmpeg_client import CONTAINER_TEMP_ROOT, remove_container_path

    cdir = CONTAINER_TEMP_ROOT / f"it-{uuid.uuid4().hex[:12]}"
    yield cdir
    try:
        remove_container_path(cdir)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_container
class TestDockerCp:
    def test_copy_to_and_from_container(
        self, work_dir: Path, container_dir: PurePosixPath
    ) -> None:
        from src.ffmpeg_client import copy_from_container, copy_to_container

        src_file = work_dir / "hello.txt"
        src_file.write_text("docker-cp round trip", encoding="utf-8")

        copied = copy_to_container(src_file, container_dir)
        assert copied == container_dir / "hello.txt"

        back_dir = work_dir / "back"
        result = copy_from_container(copied, back_dir)
        assert result.read_text(encoding="utf-8") == "docker-cp round trip"

    def test_copy_from_missing_file_raises(
        self, work_dir: Path, container_dir: PurePosixPath
    ) -> None:
        from src.ffmpeg_client import copy_from_container

        with pytest.raises((FileNotFoundError, RuntimeError)):
            copy_from_container(container_dir / "does-not-exist.bin", work_dir)


@skip_if_no_container
class TestProbeDuration:
    def test_probe_duration_returns_float(
        self, test_video: Path, container_dir: PurePosixPath
    ) -> None:
        from src.ffmpeg_client import copy_to_container, probe_duration

        c_video = copy_to_container(test_video, container_dir)
        duration = probe_duration(str(c_video))
        assert isinstance(duration, float)
        assert 4.0 <= duration <= 6.0  # generated 5-second video


@skip_if_no_container
class TestExtractAudio:
    def test_extract_audio_creates_file(
        self, work_dir: Path, test_video: Path, container_dir: PurePosixPath
    ) -> None:
        from src.ffmpeg_client import (
            copy_from_container,
            copy_to_container,
            extract_audio,
        )

        c_video = copy_to_container(test_video, container_dir)
        c_audio = container_dir / "test_audio.m4a"
        extract_audio(str(c_video), str(c_audio))

        audio_out = copy_from_container(c_audio, work_dir)
        assert audio_out.exists()
        assert audio_out.stat().st_size > 0


@skip_if_no_container
class TestBurnSubtitles:
    def test_burn_subtitles_creates_output(
        self, work_dir: Path, test_video: Path, container_dir: PurePosixPath
    ) -> None:
        from src.ffmpeg_client import (
            burn_subtitles,
            copy_from_container,
            copy_to_container,
        )
        from src.subtitle import Segment, write_srt

        srt_path = work_dir / "test.srt"
        write_srt([Segment(0.0, 2.0, "Hello World")], srt_path)

        c_video = copy_to_container(test_video, container_dir)
        c_srt = copy_to_container(srt_path, container_dir)
        c_output = container_dir / "test_output.mp4"

        burn_subtitles(str(c_video), str(c_srt), str(c_output), fonts_dir=None)

        output = copy_from_container(c_output, work_dir)
        assert output.exists()
        assert output.stat().st_size > 0
