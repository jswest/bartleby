"""Tests for the session state layer (bartleby.session)."""

from __future__ import annotations

import pytest

import bartleby.config
import bartleby.db.connection
import bartleby.project
import bartleby.session as session_mod
from bartleby.db.connection import open_db


@pytest.fixture
def project(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(bartleby.config, "BARTLEBY_DIR", tmp_path)
    monkeypatch.setattr(bartleby.config, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(bartleby.project, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.db.connection, "PROJECTS_DIR", projects)

    bartleby.project.create_project("alpha")
    return "alpha"


def test_start_session_creates_row_and_writes_pointer(project):
    info = session_mod.start_session(project)
    assert info["session_id"] >= 1
    assert "-" in info["name"]
    assert info["memory_enabled"] is True
    assert info["ended_at"] is None

    assert session_mod.read_active_session_id(project) == info["session_id"]

    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT session_id, name, memory_enabled FROM sessions"
        ).fetchone()
        assert row == (info["session_id"], info["name"], 1)
    finally:
        conn.close()


def test_start_session_no_memory_writes_zero(project):
    info = session_mod.start_session(project, memory_enabled=False)
    assert info["memory_enabled"] is False

    conn = open_db(project)
    try:
        memory = conn.cursor().execute(
            "SELECT memory_enabled FROM sessions WHERE session_id = ?",
            (info["session_id"],),
        ).fetchone()[0]
        assert memory == 0
    finally:
        conn.close()


def test_get_current_session_reads_pointer(project):
    started = session_mod.start_session(project)
    current = session_mod.get_current_session(project)
    assert current is not None
    assert current["session_id"] == started["session_id"]
    assert current["name"] == started["name"]


def test_get_current_session_returns_none_when_no_pointer(project):
    assert session_mod.get_current_session(project) is None


def test_start_session_replaces_active_pointer(project):
    first = session_mod.start_session(project)
    second = session_mod.start_session(project, memory_enabled=False)

    assert second["session_id"] != first["session_id"]
    assert session_mod.read_active_session_id(project) == second["session_id"]

    # Both rows still exist in the DB; only the pointer changed.
    conn = open_db(project)
    try:
        n = conn.cursor().execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert n == 2
    finally:
        conn.close()


def test_end_active_session_clears_pointer_and_stamps_ended_at(project):
    started = session_mod.start_session(project)
    ended = session_mod.end_active_session(project)
    assert ended is not None
    assert ended["session_id"] == started["session_id"]
    assert ended["ended_at"] is not None

    assert session_mod.read_active_session_id(project) is None
    assert session_mod.get_current_session(project) is None


def test_end_active_session_when_none_active_returns_none(project):
    assert session_mod.end_active_session(project) is None


def test_get_current_handles_stale_pointer(project):
    started = session_mod.start_session(project)
    # Manually delete the row out from under the pointer.
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (started["session_id"],),
        )
    finally:
        conn.close()

    assert session_mod.get_current_session(project) is None
    # Stale pointer should have been cleaned up.
    assert session_mod.read_active_session_id(project) is None


def test_ensure_active_session_returns_existing_id(project):
    started = session_mod.start_session(project)
    assert session_mod.ensure_active_session(project) == started["session_id"]


def test_ensure_active_session_auto_creates_when_missing(project):
    assert session_mod.read_active_session_id(project) is None
    new_id = session_mod.ensure_active_session(project)
    assert new_id >= 1

    info = session_mod.get_current_session(project)
    assert info is not None
    assert info["session_id"] == new_id
    assert info["memory_enabled"] is True   # default for auto-created


def test_ensure_active_session_replaces_stale_pointer(project):
    started = session_mod.start_session(project)
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (started["session_id"],),
        )
    finally:
        conn.close()

    new_id = session_mod.ensure_active_session(project)
    # The pointer must point at a row that actually exists; SQLite may
    # reuse the prior session_id, so we don't assert it differs.
    assert session_mod.read_active_session_id(project) == new_id
    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (new_id,)
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


def test_ensure_named_session_creates_memory_enabled_row(project):
    sid = session_mod.ensure_named_session(project, "web-reader")
    assert sid >= 1

    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT name, memory_enabled FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
        assert row == ("web-reader", 1)
    finally:
        conn.close()


def test_ensure_named_session_is_idempotent(project):
    first = session_mod.ensure_named_session(project, "web-reader")
    second = session_mod.ensure_named_session(project, "web-reader")
    assert first == second

    conn = open_db(project)
    try:
        n = conn.cursor().execute(
            "SELECT COUNT(*) FROM sessions WHERE name = ?", ("web-reader",)
        ).fetchone()[0]
        assert n == 1
    finally:
        conn.close()


def test_ensure_named_session_never_touches_active_pointer(project):
    agent = session_mod.start_session(project)
    # The web caller pins its own session...
    web = session_mod.ensure_named_session(project, "web-reader")
    assert web != agent["session_id"]
    # ...without disturbing the agent's active-session pointer.
    assert session_mod.read_active_session_id(project) == agent["session_id"]


def test_generate_name_is_kebab_pair():
    name = session_mod.generate_name()
    a, _, b = name.partition("-")
    assert a and b
    assert "-" not in b
    assert name.islower()


def test_start_session_retries_on_name_collision(project, monkeypatch):
    # Force the generator to return the same name twice, then a unique one.
    names = iter(["same-name", "same-name", "fresh-name"])
    monkeypatch.setattr(session_mod, "generate_name", lambda: next(names))

    first = session_mod.start_session(project)
    second = session_mod.start_session(project)
    assert first["name"] == "same-name"
    assert second["name"] == "fresh-name"
