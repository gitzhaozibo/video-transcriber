"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv():
    """Never load the developer's real ``.env`` during tests.

    ``app.py`` (run by the E2E AppTest) and ``scripts/compare_engines.py`` call
    :func:`src.env.load_env`; without this, real keys and settings such as
    ``KEEP_TEMP_FILES`` would leak into ``os.environ`` for later tests.
    ``tests/unit/test_env.py`` imports the real function directly, so it is
    unaffected.
    """
    with patch("src.env.load_env", return_value=False):
        yield
