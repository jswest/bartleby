"""The ``BARTLEBY_HOME`` env override for Bartleby's on-disk state tree (GH-0393).

One env var relocates the *entire* tree — projects, config, scratch — so a test,
CI run, or a coding agent in an isolated worktree never touches the developer's
live ``~/.bartleby`` corpora. The override must resolve *lazily* (at call time),
or an agent that sets the env after importing bartleby would be silently ignored
— the exact failure that let a verification script clobber the live namespace.
"""

from __future__ import annotations

from pathlib import Path

from bartleby import config


def test_override_relocates_whole_tree(tmp_path, monkeypatch):
    home = tmp_path / "sandbox"
    monkeypatch.setenv("BARTLEBY_HOME", str(home))
    assert config.bartleby_dir() == home
    assert config.projects_dir() == home / "projects"
    assert config.config_path() == home / "config.yaml"
    assert config.scratch_dir() == home / "tmp"


def test_resolved_lazily_after_env_set(tmp_path, monkeypatch):
    # The crux of GH-0393: read at call time, so a change *after* import takes
    # effect. A module-level constant would have frozen the path at import and
    # silently ignored both of these.
    monkeypatch.setenv("BARTLEBY_HOME", str(tmp_path / "a"))
    assert config.bartleby_dir() == tmp_path / "a"
    monkeypatch.setenv("BARTLEBY_HOME", str(tmp_path / "b"))
    assert config.bartleby_dir() == tmp_path / "b"


def test_override_expands_user(monkeypatch):
    monkeypatch.setenv("BARTLEBY_HOME", "~/some-bartleby-sandbox")
    assert config.bartleby_dir() == Path.home() / "some-bartleby-sandbox"


def test_falls_back_to_home_when_unset(monkeypatch):
    monkeypatch.delenv("BARTLEBY_HOME", raising=False)
    # Read-only assertion: resolve the path, never create anything under it.
    assert config.bartleby_dir() == Path.home() / ".bartleby"
