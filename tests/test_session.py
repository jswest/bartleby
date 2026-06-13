"""Tests for the session state layer (bartleby.session)."""

from __future__ import annotations

import os

import pytest

import bartleby.project
import bartleby.session as session_mod
from bartleby.db.connection import open_db


@pytest.fixture
def project():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
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


def test_write_active_session_id_is_atomic_no_partial_read(project, monkeypatch):
    # write_active_session_id must publish via an atomic rename: a concurrent
    # reader interleaved mid-write sees either the old id or the new one in
    # full, never a truncated line. We intercept os.replace to run a read at
    # exactly the moment the new content exists in a temp file but the pointer
    # hasn't been swapped yet.
    session_mod.write_active_session_id(project, 11)

    observed = []
    orig_replace = os.replace

    def spy_replace(src, dst):
        # Mid-write: reader still sees the OLD committed value, never a partial.
        observed.append(session_mod.read_active_session_id(project))
        orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    session_mod.write_active_session_id(project, 22)

    assert observed == [11]  # old value still visible until the rename lands
    assert session_mod.read_active_session_id(project) == 22


def test_write_active_session_id_leaves_no_tmp_file(project):
    session_mod.write_active_session_id(project, 7)
    pointer = bartleby.project.get_project_dir(project) / ".active_session"
    assert not pointer.with_name(pointer.name + ".tmp").exists()
    assert session_mod.read_active_session_id(project) == 7


def test_write_active_session_id_concurrent_writers_no_race(project):
    # Regression for #553: a shared ".active_session.tmp" let concurrent writers
    # clobber one temp and race on the rename — the second os.replace raised
    # FileNotFoundError. Unique temps fix it: many overlapping writers all
    # succeed, the pointer lands on one written id, and no temp is left behind.
    import threading

    ids = list(range(1, 51))
    errors: list[Exception] = []

    def writer(sid: int) -> None:
        try:
            for _ in range(10):
                session_mod.write_active_session_id(project, sid)
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(sid,)) for sid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert session_mod.read_active_session_id(project) in ids
    pdir = bartleby.project.get_project_dir(project)
    assert not list(pdir.glob(".active_session.*.tmp"))


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


# ---------- model / harness provenance (issue #62) ----------


def test_detect_harness_recognizes_claude_code(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    assert session_mod.detect_harness() == "claude-code"


def test_detect_harness_unknown_is_none(monkeypatch):
    monkeypatch.delenv("CLAUDECODE", raising=False)
    assert session_mod.detect_harness() is None


def test_start_session_records_explicit_provenance(project, monkeypatch):
    monkeypatch.delenv("CLAUDECODE", raising=False)
    info = session_mod.start_session(project, model="claude-opus-4-8", harness="goose")
    assert info["model"] == "claude-opus-4-8"
    assert info["harness"] == "goose"

    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT model, harness FROM sessions WHERE session_id = ?",
            (info["session_id"],),
        ).fetchone()
        assert row == ("claude-opus-4-8", "goose")
    finally:
        conn.close()


def test_start_session_autodetects_harness_when_omitted(project, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    info = session_mod.start_session(project)
    assert info["harness"] == "claude-code"
    # Model is rarely in the environment, so it stays null without an explicit value.
    assert info["model"] is None


def test_start_session_explicit_harness_overrides_detection(project, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    info = session_mod.start_session(project, harness="pi")
    assert info["harness"] == "pi"


def test_set_session_provenance_updates_only_passed_fields(project, monkeypatch):
    monkeypatch.delenv("CLAUDECODE", raising=False)
    started = session_mod.start_session(project)
    assert started["model"] is None and started["harness"] is None

    after_model = session_mod.set_session_provenance(project, model="qwen3.6:35b-mlx")
    assert after_model["model"] == "qwen3.6:35b-mlx"
    assert after_model["harness"] is None  # untouched

    after_harness = session_mod.set_session_provenance(project, harness="ollama-cli")
    assert after_harness["model"] == "qwen3.6:35b-mlx"  # preserved
    assert after_harness["harness"] == "ollama-cli"


def test_set_session_provenance_no_active_returns_none(project):
    assert session_mod.set_session_provenance(project, model="x") is None


# --- run_key: one run per conversation (#547) ---------------------------------


def test_start_session_persists_run_key(project):
    info = session_mod.start_session(project, run_key="run-abc")
    assert info["run_key"] == "run-abc"

    conn = open_db(project)
    try:
        stored = conn.cursor().execute(
            "SELECT run_key FROM sessions WHERE session_id = ?",
            (info["session_id"],),
        ).fetchone()[0]
        assert stored == "run-abc"
    finally:
        conn.close()


def test_start_session_run_key_defaults_null(project):
    info = session_mod.start_session(project)
    assert info["run_key"] is None


def test_ensure_session_by_run_key_creates_then_reuses(project):
    first = session_mod.ensure_session_by_run_key(project, "run-1")
    again = session_mod.ensure_session_by_run_key(project, "run-1")
    assert first == again  # same key → same run, not a second row

    conn = open_db(project)
    try:
        count = conn.cursor().execute(
            "SELECT COUNT(*) FROM sessions WHERE run_key = ?", ("run-1",),
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_ensure_session_by_run_key_distinct_keys_distinct_runs(project):
    a = session_mod.ensure_session_by_run_key(project, "run-A")
    b = session_mod.ensure_session_by_run_key(project, "run-B")
    assert a != b  # two conversations never collide


def test_ensure_session_by_run_key_writes_active_marker(project):
    sid = session_mod.ensure_session_by_run_key(project, "run-mark")
    # The marker stays warm so a later call that forgets --run still resolves
    # the most recent run.
    assert session_mod.read_active_session_id(project) == sid


def test_ensure_session_by_run_key_created_run_is_memory_enabled(project):
    sid = session_mod.ensure_session_by_run_key(project, "run-mem")
    conn = open_db(project)
    try:
        info = session_mod._fetch_session(conn, sid)
    finally:
        conn.close()
    assert info["memory_enabled"] is True


def test_run_echo_marks_self_reported_model(project):
    info = session_mod.start_session(project, run_key="run-llm", model="opus")
    conn = open_db(project)
    try:
        echo = session_mod.run_echo(conn, info["session_id"])
    finally:
        conn.close()
    assert echo["run_key"] == "run-llm"
    assert echo["model"] == "opus"
    assert echo["session_name"] == info["name"]
    assert echo["model_set_by_llm"] is True


def test_run_echo_no_llm_label_without_run_key(project):
    # A CLI/web session (no run_key) carrying a model is NOT an LLM self-report.
    info = session_mod.start_session(project, model="opus")
    conn = open_db(project)
    try:
        echo = session_mod.run_echo(conn, info["session_id"])
    finally:
        conn.close()
    assert echo["run_key"] is None
    assert echo["model_set_by_llm"] is False


def test_run_echo_no_llm_label_when_model_absent(project):
    info = session_mod.start_session(project, run_key="run-nomodel")
    conn = open_db(project)
    try:
        echo = session_mod.run_echo(conn, info["session_id"])
    finally:
        conn.close()
    assert echo["model_set_by_llm"] is False
