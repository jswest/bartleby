"""Pytest-wide fixtures.

Skill scripts live in ``bartleby.skill_scripts`` and are reachable via normal
package imports; nothing path-y needed here.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _isolate_bartleby_home(tmp_path, monkeypatch):
    """Point ``BARTLEBY_HOME`` at a per-test sandbox so no test can read or write
    the developer's live ``~/.bartleby`` corpora.

    This is the suite-wide, fail-safe-by-default backstop that replaces the ~9
    copies of per-fixture ``PROJECTS_DIR`` monkeypatching — which protected only
    the files that remembered to opt in and left every new test file exposed (the
    hole behind GH-0393). ``tmp_path`` is used as the root directly, so
    ``projects_dir()`` / ``config_path()`` / ``scratch_dir()`` resolve to
    ``tmp_path/{projects,config.yaml,tmp}`` — the exact layout the old fixtures
    built by hand, so existing path assertions still hold. A test that needs a
    different home can ``monkeypatch.setenv`` over this; one exercising the
    real-home fallback must only *read* the resolved path, never write.
    """
    monkeypatch.setenv("BARTLEBY_HOME", str(tmp_path))
    from bartleby import config

    assert config.bartleby_dir() == tmp_path, "BARTLEBY_HOME override not honored"
    yield


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
