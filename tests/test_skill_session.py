"""Tests for the `session new` skill script (one run per conversation, #547)."""

from __future__ import annotations

import json

import pytest

from bartleby.db.connection import open_db
from bartleby.skill_scripts import session as session_script
from tests._skill_fixtures import project_env  # noqa: F401


def test_session_new_mints_run_and_returns_it(project_env, capsys):
    session_script.main(["new", "--project", project_env])
    out = json.loads(capsys.readouterr().out)
    assert out["created"] is True
    assert out["run"]["run_key"]  # a non-empty UUID the agent will carry
    assert out["run"]["memory_enabled"] is True
    assert out["run"]["model"] is None
    assert out["run"]["model_set_by_llm"] is False


def test_session_new_distinct_each_call(project_env, capsys):
    session_script.main(["new", "--project", project_env])
    first = json.loads(capsys.readouterr().out)
    session_script.main(["new", "--project", project_env])
    second = json.loads(capsys.readouterr().out)
    # New conversation → new run: distinct id AND distinct session row.
    assert first["run"]["run_key"] != second["run"]["run_key"]
    assert first["run"]["session_id"] != second["run"]["session_id"]


def test_session_new_records_self_reported_model(project_env, capsys):
    session_script.main(["new", "--project", project_env, "--model", "opus"])
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["model"] == "opus"
    assert out["run"]["model_set_by_llm"] is True  # a claim, "Set by LLM"


def test_session_new_no_memory(project_env, capsys):
    session_script.main(["new", "--project", project_env, "--no-memory"])
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["memory_enabled"] is False


def test_session_new_writes_audit_row(project_env, capsys):
    session_script.main(["new", "--project", project_env])
    json.loads(capsys.readouterr().out)  # drain
    conn = open_db(project_env)
    try:
        count = conn.cursor().execute(
            "SELECT COUNT(*) FROM audit_logs WHERE tool_name = 'session_new'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_session_new_requires_subcommand(project_env, capsys):
    with pytest.raises(SystemExit) as exc:
        session_script.main(["--project", project_env])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "USAGE_ERROR"
