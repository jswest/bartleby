"""Tests for the release helper (scripts/release.py) and `bartleby --version`.

The release script lives under scripts/ (not an installed package), so it's
loaded by path. The pure functions — version arithmetic, schema parsing, the
drift guard, and notes assembly — are exercised here, plus the recovery path for
a `gh` publish that fails after the tag is already pushed (subprocess/git/gh
calls are mocked; nothing is tagged, pushed, or released for real).
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_RELEASE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "release.py"
_spec = importlib.util.spec_from_file_location("release", _RELEASE_PATH)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


SCHEMA_TEMPLATE = '''\
"""schema."""

SCHEMA_VERSION = {version}

EMBEDDING_DIM = 768

DDL = """{ddl}"""
'''


def _schema_source(version: int, ddl: str = "CREATE TABLE meta (k TEXT);") -> str:
    return SCHEMA_TEMPLATE.format(version=version, ddl=ddl)


# --- compute_next_version --------------------------------------------------

def test_baseline_when_no_prior_tag():
    assert release.compute_next_version(7, None) == "0.7.0"


def test_patch_bump_when_schema_unchanged():
    assert release.compute_next_version(7, "v0.7.2") == "0.7.3"


def test_minor_tracks_schema_and_resets_patch():
    assert release.compute_next_version(8, "v0.7.2") == "0.8.0"


def test_minor_jumps_when_schema_skips():
    # Two schema bumps between releases -> jump straight to the new minor.
    assert release.compute_next_version(9, "v0.7.0") == "0.9.0"


def test_major_is_carried_forward_unchanged():
    assert release.compute_next_version(8, "v1.7.4") == "1.8.0"
    assert release.compute_next_version(7, "v1.7.4") == "1.7.5"


def test_parse_version_tag():
    assert release.parse_version_tag("v0.7.2") == (0, 7, 2)
    assert release.parse_version_tag("0.7.2") == (0, 7, 2)
    with pytest.raises(ValueError):
        release.parse_version_tag("v0.7")


# --- schema_moved ----------------------------------------------------------

def test_schema_moved():
    assert release.schema_moved(8, "v0.7.0") is True
    assert release.schema_moved(7, "v0.7.3") is False
    assert release.schema_moved(7, None) is False


# --- parse_schema_module ---------------------------------------------------

def test_parse_schema_module_extracts_version_and_ddl():
    version, ddl = release.parse_schema_module(_schema_source(7, "CREATE TABLE x (a);"))
    assert version == 7
    assert ddl == "CREATE TABLE x (a);"


def test_parse_schema_module_raises_without_version():
    with pytest.raises(ValueError):
        release.parse_schema_module('DDL = """x"""\n')


# --- check_drift -----------------------------------------------------------

def test_no_drift_when_nothing_changed():
    src = _schema_source(7)
    assert release.check_drift(src, src) is None


def test_no_drift_when_ddl_changed_and_version_bumped():
    old = _schema_source(7, "CREATE TABLE a (x);")
    new = _schema_source(8, "CREATE TABLE a (x); CREATE TABLE b (y);")
    assert release.check_drift(old, new) is None


def test_drift_when_ddl_changed_but_version_static():
    old = _schema_source(7, "CREATE TABLE a (x);")
    new = _schema_source(7, "CREATE TABLE a (x); CREATE TABLE b (y);")
    problem = release.check_drift(old, new)
    assert problem is not None
    assert "SCHEMA_VERSION" in problem


def test_version_bump_without_ddl_change_is_allowed():
    old = _schema_source(7)
    new = _schema_source(8)
    assert release.check_drift(old, new) is None


# --- build_release_notes ---------------------------------------------------

def test_notes_include_commits():
    notes = release.build_release_notes(
        ["Fix a thing", "Add another"], schema_from=None, schema_to=None,
    )
    assert "- Fix a thing" in notes
    assert "- Add another" in notes
    assert "⚠️" not in notes  # no re-ingest banner when the schema didn't move


def test_notes_reingest_banner_on_breaking_bump(monkeypatch):
    # No _UPGRADES entry for the 8 -> 9 step: breaking bump, re-ingest.
    monkeypatch.setattr(release, "_UPGRADES", {})
    notes = release.build_release_notes(
        ["Big change"], schema_from=8, schema_to=9,
    )
    assert "⚠️" in notes
    assert "8 → 9" in notes
    assert "re-ingested" in notes
    assert "project upgrade" not in notes


def test_notes_upgrade_banner_on_additive_bump(monkeypatch):
    # Every crossed step (8 -> 9) has a chain entry: upgrade in place, no re-ingest.
    monkeypatch.setattr(release, "_UPGRADES", {8: object()})
    notes = release.build_release_notes(
        ["Big change"], schema_from=8, schema_to=9,
    )
    assert "⚠️" in notes
    assert "8 → 9" in notes
    assert "bartleby project upgrade <name>" in notes
    assert "re-ingested" not in notes


# --- upgrade_covers --------------------------------------------------------

def test_upgrade_covers_single_step_present(monkeypatch):
    monkeypatch.setattr(release, "_UPGRADES", {7: object()})
    assert release.upgrade_covers(7, 8) is True


def test_upgrade_covers_multi_step_all_present(monkeypatch):
    monkeypatch.setattr(release, "_UPGRADES", {5: object(), 6: object(), 7: object()})
    assert release.upgrade_covers(5, 8) is True


def test_upgrade_covers_missing_step_is_breaking(monkeypatch):
    monkeypatch.setattr(release, "_UPGRADES", {5: object(), 7: object()})  # 6 missing
    assert release.upgrade_covers(5, 8) is False


def test_upgrade_covers_no_entry_is_breaking(monkeypatch):
    monkeypatch.setattr(release, "_UPGRADES", {})
    assert release.upgrade_covers(8, 9) is False


def test_upgrade_covers_non_forward_is_false(monkeypatch):
    monkeypatch.setattr(release, "_UPGRADES", {8: object()})
    assert release.upgrade_covers(9, 9) is False
    assert release.upgrade_covers(9, 8) is False


def test_notes_baseline_when_no_commits():
    notes = release.build_release_notes(
        [], schema_from=None, schema_to=None,
    )
    assert "Baseline release" in notes


# --- gh publish recovery ---------------------------------------------------

def test_gh_recovery_message_writes_notes_and_returns_command(tmp_path, monkeypatch):
    # Redirect the scratch root so the test never touches the real repo tree.
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    msg = release.gh_recovery_message("v0.9.7", "## Changes\n- A thing\n")

    notes_file = tmp_path / ".claude" / "scratch" / "release-notes-v0.9.7.md"
    assert notes_file.read_text() == "## Changes\n- A thing\n"
    # The message hands back the exact resume command and points at the file.
    assert f"gh release create v0.9.7 --title v0.9.7 --notes-file {notes_file}" in msg
    assert "do NOT re-run this script" in msg
    assert str(notes_file) in msg


def test_main_publish_failure_is_recoverable_and_nonzero(tmp_path, monkeypatch, capsys):
    # Mock every git/gh side effect: the run reaches the publish step without
    # tagging, pushing, or releasing for real, and `gh release create` fails.
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release, "last_release_tag", lambda: "v0.9.6")
    monkeypatch.setattr(release, "schema_source_at", lambda ref: _schema_source(9))
    monkeypatch.setattr(release, "commits_since", lambda ref: ["Fix a thing"])
    monkeypatch.setattr(release, "working_tree_dirty", lambda: False)
    monkeypatch.setattr(release, "current_branch", lambda: "main")

    git_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(release, "_git", lambda *a: git_calls.append(a) or "")

    # schema_version is read from REPO_ROOT/SCHEMA_PATH — stage a schema 9 file.
    schema_file = tmp_path / release.SCHEMA_PATH
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    schema_file.write_text(_schema_source(9))

    def fake_run(cmd, *args, **kwargs):
        # Only the gh publish goes through subprocess.run in the push path.
        assert cmd[:3] == ["gh", "release", "create"]
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    rc = release.main(["--push"])

    assert rc == 1  # non-zero exit on the failed publish
    # The tag was pushed before gh ran (the failure window the recovery covers).
    assert ("tag", "-a", "v0.9.7", "-m", "Release v0.9.7") in git_calls
    assert ("push", "origin", "v0.9.7") in git_calls

    err = capsys.readouterr().err
    notes_file = tmp_path / ".claude" / "scratch" / "release-notes-v0.9.7.md"
    assert "gh release create v0.9.7" in err
    assert "--notes-file" in err
    assert notes_file.exists()


def test_main_publish_recoverable_when_gh_not_installed(tmp_path, monkeypatch, capsys):
    # A missing `gh` binary is the most likely first-time failure: subprocess.run
    # raises FileNotFoundError, not CalledProcessError. It hits the same
    # tag-pushed-but-no-Release window, so it must be just as recoverable — not a
    # traceback that loses the notes.
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release, "last_release_tag", lambda: "v0.9.6")
    monkeypatch.setattr(release, "schema_source_at", lambda ref: _schema_source(9))
    monkeypatch.setattr(release, "commits_since", lambda ref: ["Fix a thing"])
    monkeypatch.setattr(release, "working_tree_dirty", lambda: False)
    monkeypatch.setattr(release, "current_branch", lambda: "main")
    monkeypatch.setattr(release, "_git", lambda *a: "")

    schema_file = tmp_path / release.SCHEMA_PATH
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    schema_file.write_text(_schema_source(9))

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'gh'")

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    rc = release.main(["--push"])

    assert rc == 1
    err = capsys.readouterr().err
    notes_file = tmp_path / ".claude" / "scratch" / "release-notes-v0.9.7.md"
    assert "gh release create v0.9.7" in err
    assert notes_file.exists()


# --- bartleby --version ----------------------------------------------------

def test_version_flag_prints_and_exits_zero(capsys, monkeypatch):
    from bartleby.cli import main

    monkeypatch.setattr("sys.argv", ["bartleby", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "bartleby" in capsys.readouterr().out
