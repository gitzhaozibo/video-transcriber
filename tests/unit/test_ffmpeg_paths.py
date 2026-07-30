"""Unit tests for src.ffmpeg_client path-conversion utilities.

Docker SDK calls are NOT made; only the pure path-conversion logic is tested.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import pytest


# We patch HOST_SHARED_DIR before importing to avoid side effects from env.
FAKE_SHARED = Path("/fake/shared").resolve()


def _import_with_fake_shared():
    """Re-import ffmpeg_client with a patched HOST_SHARED_DIR."""
    import importlib
    import sys

    # Remove cached module so we get a fresh import with our patch.
    sys.modules.pop("src.ffmpeg_client", None)
    with patch.dict(os.environ, {"SHARED_DIR": str(FAKE_SHARED)}):
        import src.ffmpeg_client as mod
        return mod


class TestHostToContainer:
    def setup_method(self) -> None:
        self.mod = _import_with_fake_shared()

    def test_simple_path(self) -> None:
        host = FAKE_SHARED / "jobs" / "abc" / "video.mp4"
        result = self.mod.host_to_container(host)
        assert result == "/work/jobs/abc/video.mp4"

    def test_nested_path(self) -> None:
        host = FAKE_SHARED / "a" / "b" / "c" / "file.m4a"
        result = self.mod.host_to_container(host)
        assert result == "/work/a/b/c/file.m4a"

    def test_outside_shared_raises(self) -> None:
        host = Path("/tmp/evil.mp4")
        with pytest.raises(ValueError, match="not inside shared directory"):
            self.mod.host_to_container(host)

    def test_exact_shared_root(self) -> None:
        host = FAKE_SHARED / "file.txt"
        result = self.mod.host_to_container(host)
        assert result == "/work/file.txt"


class TestContainerToHost:
    def setup_method(self) -> None:
        self.mod = _import_with_fake_shared()

    def test_simple_path(self) -> None:
        result = self.mod.container_to_host("/work/jobs/abc/audio.m4a")
        assert result == FAKE_SHARED / "jobs" / "abc" / "audio.m4a"

    def test_not_under_work_raises(self) -> None:
        with pytest.raises(ValueError, match="not under"):
            self.mod.container_to_host("/tmp/other")

    def test_nested(self) -> None:
        result = self.mod.container_to_host("/work/a/b/c.srt")
        assert result == FAKE_SHARED / "a" / "b" / "c.srt"


class TestRoundTrip:
    def setup_method(self) -> None:
        self.mod = _import_with_fake_shared()

    def test_host_to_container_and_back(self) -> None:
        host = FAKE_SHARED / "jobs" / "uuid-1234" / "video.mp4"
        container = self.mod.host_to_container(host)
        restored = self.mod.container_to_host(container)
        assert restored == host
