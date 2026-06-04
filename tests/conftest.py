"""Pytest-wide fixtures.

Skill scripts live in ``bartleby.skill_scripts`` and are reachable via normal
package imports; nothing path-y needed here.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _canonical_tmpdir():
    """Point ``$TMPDIR`` at a symlink-resolved path so the tesseract subprocess
    can read pytesseract's temp images.

    macOS's ``/tmp`` is a symlink to ``/private/tmp``, and the homebrew
    tesseract's leptonica image loader fails to open files addressed *through*
    that symlink ("image file not found"). pytesseract serializes the image to
    a temp file under ``$TMPDIR`` and shells out to ``tesseract``, so a
    ``/tmp``-based ``$TMPDIR`` (e.g. a CI/sandbox harness that sets
    ``TMPDIR=/tmp/...``) silently breaks OCR — and pytesseract then masks the
    failure with a ``UnicodeDecodeError``. Resolving ``$TMPDIR`` with
    ``realpath`` sidesteps it; it's a no-op where ``$TMPDIR`` is already
    canonical (Linux ``/tmp``, macOS ``/var/folders/...``). See issue #43.
    """
    original = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = os.path.realpath(tempfile.gettempdir())
    tempfile.tempdir = None  # drop the cached temp dir so $TMPDIR is re-read
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = original
        tempfile.tempdir = None
