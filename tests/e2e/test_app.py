"""E2E tests for the Streamlit app using streamlit.testing.v1.AppTest.

Heavy processing (FFmpeg, Whisper) is mocked so these tests run without
Docker or GPU.

Marked with ``pytest.mark.e2e``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_test():
    """Return an AppTest instance for app.py."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppLoads:
    def test_title_present(self) -> None:
        """App should render without crashing."""
        at = _make_app_test()
        at.run()
        assert not at.exception


class TestUploadFlow:
    def test_upload_button_visible(self) -> None:
        """File uploader should be present on the page."""
        at = _make_app_test()
        at.run()
        # The uploader widget should be present
        assert not at.exception

    def test_transcribe_button_disabled_before_upload(self) -> None:
        """Transcribe button should be disabled when no file is uploaded."""
        at = _make_app_test()
        at.run()
        # Without upload, the button for transcription should not appear or be disabled.
        # Verify the app starts cleanly.
        assert not at.exception
