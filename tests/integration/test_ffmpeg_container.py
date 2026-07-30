"""Integration tests for FFmpeg container operations.

These tests require a running ``ffmpeg-worker`` Docker container.
They are marked with ``pytest.mark.integration`` and skipped automatically
when the container is not available.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
def shared_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary shared directory and patch HOST_SHARED_DIR."""
    d = tmp_path_factory.mktemp("shared")
    return d


@pytest.fixture(scope="module")
def test_video(shared_dir: Path) -> Path:
    """Generate a 5-second test video using the host ffmpeg or container."""
    video_path = shared_dir / "test_video.mp4"
    # Use ffmpeg to generate a silent 5-second video with a test tone
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_container
class TestExtractAudio:
    def test_extract_audio_creates_file(self, shared_dir: Path, test_video: Path) -> None:
        import src.ffmpeg_client as client

        orig_shared = client.HOST_SHARED_DIR
        client.HOST_SHARED_DIR = shared_dir
        try:
            audio_out = shared_dir / "test_audio.m4a"
            client.extract_audio(test_video, audio_out)
            assert audio_out.exists()
            assert audio_out.stat().st_size > 0
        finally:
            client.HOST_SHARED_DIR = orig_shared


@skip_if_no_container
class TestProbeDuration:
    def test_probe_duration_returns_float(self, shared_dir: Path, test_video: Path) -> None:
        import src.ffmpeg_client as client

        orig_shared = client.HOST_SHARED_DIR
        client.HOST_SHARED_DIR = shared_dir
        try:
            duration = client.probe_duration(test_video)
            assert isinstance(duration, float)
            assert 4.0 <= duration <= 6.0  # generated 5-second video
        finally:
            client.HOST_SHARED_DIR = orig_shared


@skip_if_no_container
class TestBurnSubtitles:
    def test_burn_subtitles_creates_output(
        self, shared_dir: Path, test_video: Path
    ) -> None:
        import src.ffmpeg_client as client
        from src.subtitle import Segment, write_srt

        orig_shared = client.HOST_SHARED_DIR
        client.HOST_SHARED_DIR = shared_dir
        try:
            srt_path = shared_dir / "test.srt"
            write_srt([Segment(0.0, 2.0, "Hello World")], srt_path)

            output_path = shared_dir / "test_output.mp4"
            client.burn_subtitles(test_video, srt_path, output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        finally:
            client.HOST_SHARED_DIR = orig_shared
