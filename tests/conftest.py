"""Pytest-wide fixtures.

Skill scripts live in ``bartleby.skill_scripts`` and are reachable via normal
package imports; nothing path-y needed here.
"""

from __future__ import annotations

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
