"""Tests for `bartleby logs` rendering."""

from __future__ import annotations

import pytest

import bartleby.project
from bartleby.commands import logs as logs_cmd
from bartleby.db.audit import log_call
from bartleby.db.connection import open_db
from bartleby.session import write_active_session_id


@pytest.fixture
def project():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    bartleby.project.create_project("alpha")
    return "alpha"


def _add_session(project_name, name):
    conn = open_db(project_name)
    try:
        conn.cursor().execute("INSERT INTO sessions (name) VALUES (?)", (name,))
        sid = conn.last_insert_rowid()
    finally:
        conn.close()
    return sid


def _add_log(project_name, *, session_id, tool_name, args=None, duration_ms=None):
    conn = open_db(project_name)
    try:
        log_call(
            conn,
            session_id=session_id,
            tool_name=tool_name,
            args=args,
            duration_ms=duration_ms,
        )
    finally:
        conn.close()


def test_logs_no_sessions(project, capsys):
    logs_cmd.main(session=None, limit=50, project=project)
    assert "No sessions yet" in capsys.readouterr().out


def test_logs_defaults_to_most_recent_session(project, capsys):
    older = _add_session(project, "older-sess")
    newer = _add_session(project, "newer-sess")
    _add_log(project, session_id=older, tool_name="search_old")
    _add_log(project, session_id=newer, tool_name="search_new")

    logs_cmd.main(session=None, limit=50, project=project)
    out = capsys.readouterr().out
    assert "newer-sess" in out
    assert "search_new" in out
    assert "search_old" not in out


def test_logs_prefers_active_session_over_newest(project, capsys):
    older = _add_session(project, "older-sess")
    newer = _add_session(project, "newer-sess")
    _add_log(project, session_id=older, tool_name="search_old")
    _add_log(project, session_id=newer, tool_name="search_new")
    write_active_session_id(project, older)

    logs_cmd.main(session=None, limit=50, project=project)
    out = capsys.readouterr().out
    assert "older-sess" in out
    assert "search_old" in out
    assert "search_new" not in out


def test_logs_falls_back_to_newest_when_active_id_stale(project, capsys):
    older = _add_session(project, "older-sess")
    newer = _add_session(project, "newer-sess")
    _add_log(project, session_id=older, tool_name="search_old")
    _add_log(project, session_id=newer, tool_name="search_new")
    write_active_session_id(project, 9999)  # no such session row

    logs_cmd.main(session=None, limit=50, project=project)
    out = capsys.readouterr().out
    assert "newer-sess" in out
    assert "search_new" in out
    assert "search_old" not in out


def test_logs_by_explicit_session_name(project, capsys):
    a = _add_session(project, "alpha-grove")
    b = _add_session(project, "beta-river")
    _add_log(project, session_id=a, tool_name="tool_a")
    _add_log(project, session_id=b, tool_name="tool_b")

    logs_cmd.main(session="alpha-grove", limit=50, project=project)
    out = capsys.readouterr().out
    assert "alpha-grove" in out
    assert "tool_a" in out
    assert "tool_b" not in out


def test_logs_unknown_session_exits_with_error(project, capsys):
    _add_session(project, "real-sess")
    with pytest.raises(SystemExit) as exc:
        logs_cmd.main(session="nope", limit=50, project=project)
    assert exc.value.code == 1


def test_logs_empty_session(project, capsys):
    _add_session(project, "empty-sess")
    logs_cmd.main(session="empty-sess", limit=50, project=project)
    out = capsys.readouterr().out
    assert "No logged calls" in out


def test_logs_limit_keeps_most_recent(project, capsys):
    sid = _add_session(project, "limited")
    for i in range(5):
        _add_log(project, session_id=sid, tool_name=f"tool_{i}")

    logs_cmd.main(session="limited", limit=2, project=project)
    out = capsys.readouterr().out
    # With limit=2, only the last two inserted should appear.
    assert "tool_3" in out
    assert "tool_4" in out
    assert "tool_0" not in out


def test_logs_chronological_display_order(project, capsys):
    sid = _add_session(project, "chrono")
    _add_log(project, session_id=sid, tool_name="first_call")
    _add_log(project, session_id=sid, tool_name="second_call")
    _add_log(project, session_id=sid, tool_name="third_call")

    logs_cmd.main(session="chrono", limit=50, project=project)
    out = capsys.readouterr().out
    assert out.index("first_call") < out.index("second_call") < out.index("third_call")


def test_logs_truncates_long_args(project, capsys):
    sid = _add_session(project, "long-args")
    long_args = {"query": "x" * 1000}
    _add_log(project, session_id=sid, tool_name="search", args=long_args)
    logs_cmd.main(session="long-args", limit=50, project=project)
    out = capsys.readouterr().out
    assert "…" in out
