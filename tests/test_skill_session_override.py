"""The runner honors BARTLEBY_SESSION_NAME to pin a non-agent session.

This is the seam the web UI relies on: a skill invocation made with the env
var set must run under a durable, memory-enabled session of that name and must
not read or write the project's ``.active_session`` pointer (so it never
collides with whichever session an agent has active).
"""

from __future__ import annotations

import json

from bartleby.db.connection import open_db
from bartleby.skill_scripts import list_documents, save_finding
from bartleby import session as session_mod
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
)


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


def test_empty_env_var_falls_through_to_active_session(  # noqa: F811
    project_env, monkeypatch, capsys
):
    """An empty ``BARTLEBY_SESSION_NAME`` must behave as if unset: fall through
    to the active-session path. The runner gates on truthiness (``if
    session_name:``), so this guards against a regression to ``is not None``,
    which would treat ``""`` as a pinned name and try to create a nameless
    session instead of using the agent's active one."""
    monkeypatch.setenv("BARTLEBY_SESSION_NAME", "")
    list_documents.main(["--project", project_env])
    capsys.readouterr()

    # Same outcome as the unset case: an active session was created and the
    # pointer is set; no empty-named session was pinned via the named path.
    assert session_mod.read_active_session_id(project_env) is not None
    names = [name for name, _ in _sessions(project_env)]
    assert "" not in names


def test_env_var_attributes_written_finding_to_pinned_session(  # noqa: F811
    seeded_project, monkeypatch, tmp_path, capsys
):
    """A WRITE under ``BARTLEBY_SESSION_NAME`` attributes the new row to the
    pinned session, not the ``.active_session`` pointer. Mirrors the read test
    (``test_env_var_pins_named_session``) for the mutating seam: save_finding's
    findings row carries the pinned session's id, and the agent pointer stays
    untouched."""
    monkeypatch.setenv("BARTLEBY_SESSION_NAME", "web-writer")
    project = seeded_project["project"]

    conn = open_db(project)
    try:
        cited_id = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    body_file = tmp_path / "finding.md"
    body_file.write_text(f"A pinned-session claim[^chunk:{cited_id}].", encoding="utf-8")
    save_finding.main([
        "--project", project,
        "--title", "pinned write",
        "--description", "Writing under a pinned web session.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)

    # The finding is attributed to the pinned session by name ...
    assert out["session_name"] == "web-writer"
    assert out["finding_id"].startswith("finding:")
    finding_id = int(out["finding_id"].split(":", 1)[1])

    conn = open_db(project)
    try:
        cur = conn.cursor()
        row_session_id = cur.execute(
            "SELECT session_id FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()[0]
        pinned_id = cur.execute(
            "SELECT session_id FROM sessions WHERE name = ?",
            ("web-writer",),
        ).fetchone()[0]
        assert row_session_id == pinned_id
        assert out["session_id"] == pinned_id
    finally:
        conn.close()

    # ... and the agent pointer the web path must never touch stays unset.
    assert session_mod.read_active_session_id(project) is None
