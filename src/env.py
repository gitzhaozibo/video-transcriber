"""Load settings from the project's ``.env`` file into ``os.environ``.

:func:`load_env` loads ``<project root>/.env`` (if present).  Variables that
are already set in the real environment take precedence, so a value set with
``$env:NAME = ...`` still overrides the file.

``app.py`` calls it before importing any other ``src`` module, because several
of them (e.g. :mod:`src.db`, :mod:`src.ffmpeg_client`) read ``os.environ`` at
import time.  Importing this module has no side effects, so tests never pick
up a developer's real keys from ``.env``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_FILE) -> bool:
    """Load *path* into ``os.environ`` without overriding existing variables.

    Returns True when a file was found and loaded.
    """
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements
        logger.warning(
            "%s exists but python-dotenv is not installed; run: "
            "pip install -r requirements.txt",
            path,
        )
        return False
    load_dotenv(path, override=False, encoding="utf-8")
    logger.info("Loaded settings from %s", path)
    return True
