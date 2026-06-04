"""The runner honors BARTLEBY_SESSION_NAME to pin a non-agent session.

This is the seam the web UI relies on: a skill invocation made with the env
var set must run under a durable, memory-enabled session of that name and must
not read or write the project's ``.active_session`` pointer (so it never
collides with whichever session an agent has active).
"""

from __future__ import annotations

import json

from bartleby.db.connection import open_db
from bartleby.skill_scripts import list_documents
from bartleby import session as session_mod
from tests._skill_fixtures import project_env  # noqa: F401


def _sessions(project):
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT name, memory_enabled FROM sessions ORDER BY session_id"
        ).fetchall()
    finally:
        conn.close()


def test_env_var_pins_named_session(project_env, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setenv("BARTLEBY_SESSION_NAME", "web-reader")
    list_documents.main(["--project", project_env])

    payload = json.loads(capsys.readouterr().out)
    assert "documents" in payload  # ran successfully

    assert _sessions(project_env) == [("web-reader", 1)]
    # The pointer agents use is never written by the web path.
    assert session_mod.read_active_session_id(project_env) is None


def test_without_env_var_uses_active_session(project_env, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("BARTLEBY_SESSION_NAME", raising=False)
    list_documents.main(["--project", project_env])
    capsys.readouterr()

    # Normal agent path: an active session was auto-created and the pointer set.
    assert session_mod.read_active_session_id(project_env) is not None
    names = [name for name, _ in _sessions(project_env)]
    assert "web-reader" not in names
