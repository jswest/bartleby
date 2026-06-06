"""Tests for the release helper (scripts/release.py) and `bartleby --version`.

The release script lives under scripts/ (not an installed package), so it's
loaded by path. Only the pure functions — version arithmetic, schema parsing,
the drift guard, and notes assembly — are exercised here; the git/gh side
effects are left to manual use.
"""

from __future__ import annotations

import importlib.util
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


def test_notes_banner_on_schema_bump():
    notes = release.build_release_notes(
        ["Big change"], schema_from=7, schema_to=8,
    )
    assert "⚠️" in notes
    assert "7 → 8" in notes


def test_notes_baseline_when_no_commits():
    notes = release.build_release_notes(
        [], schema_from=None, schema_to=None,
    )
    assert "Baseline release" in notes


# --- bartleby --version ----------------------------------------------------

def test_version_flag_prints_and_exits_zero(capsys, monkeypatch):
    from bartleby.cli import main

    monkeypatch.setattr("sys.argv", ["bartleby", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "bartleby" in capsys.readouterr().out
