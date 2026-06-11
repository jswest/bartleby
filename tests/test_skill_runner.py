"""Guard test for the runner transaction seam (issue #339).

``skill_runner.run(..., mutates=True)`` wraps the ``work`` call in one
``with conn:`` transaction so a mutating script commits or rolls back
atomically; the audit ``log_call`` write stays outside that transaction, so a
rolled-back mutation still records that it was attempted. Read-only scripts
(the default, ``mutates=False``) open no transaction at all.

These tests drive ``run`` directly with inline ``work`` callbacks rather than
through a real skill script, so the seam is exercised in isolation from any
particular command.
"""

from __future__ import annotations

import argparse
import json

import pytest

from bartleby.db.chunks import ChunkInput, insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.skill_runner import run
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=None)
    return p.parse_args(argv)


def _emb() -> list[float]:
    return [0.01 * i for i in range(EMBEDDING_DIM)]


def _audit_rows(project, tool_name) -> int:
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM audit_logs WHERE tool_name = ?", (tool_name,),
        ).fetchone()[0]
    finally:
        conn.close()


def test_mutating_work_that_raises_rolls_back_every_table_but_keeps_audit(
    seeded_project, capsys
):
    """A raising ``mutates=True`` work() undoes ALL its writes — across
    ``findings``, the ``chunks`` table and its fts/vec mirrors — while the audit
    row recording the failed attempt survives."""
    project = seeded_project["project"]

    written: dict = {}

    def work(*, conn, args, session_id) -> dict:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (session_id, "doomed", "hook", "body"),
        )
        finding_id = conn.last_insert_rowid()
        # Chunk insert nests its own ``with conn:`` as a savepoint under the
        # outer transaction — it must roll back too when work() raises.
        chunk_ids = insert_finding_chunks(conn, finding_id, [
            ChunkInput(text="body", embedding=_emb(), chunk_index=0),
        ])
        written["finding_id"] = finding_id
        written["chunk_ids"] = chunk_ids
        raise RuntimeError("boom after writes")

    with pytest.raises(SystemExit) as exc:
        run(
            tool_name="doomed_mutator",
            parse_args=_parse_args,
            work=work,
            argv=["--project", project],
            mutates=True,
        )
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    finding_id = written["finding_id"]
    chunk_ids = written["chunk_ids"]
    assert chunk_ids  # sanity: work() actually wrote chunks before raising

    conn = open_db(project)
    try:
        cur = conn.cursor()
        # The findings row rolled back.
        assert cur.execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id = ?", (finding_id,),
        ).fetchone()[0] == 0
        # The finding's chunks rolled back from chunks + both mirrors.
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding' "
            "AND source_id = ?",
            (finding_id,),
        ).fetchone()[0] == 0
        for cid in chunk_ids:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE rowid = ?", (cid,)
            ).fetchone()[0] == 0
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,)
            ).fetchone()[0] == 0
    finally:
        conn.close()

    # The audit row — written OUTSIDE the rolled-back transaction — survives,
    # recording that the call was attempted and failed.
    assert _audit_rows(project, "doomed_mutator") == 1


def test_readonly_work_opens_no_transaction(seeded_project):
    """A non-mutating (default) script runs work() with no open transaction —
    identical to today's behavior."""
    project = seeded_project["project"]
    seen: dict = {}

    def work(*, conn, args, session_id) -> dict:
        seen["in_transaction"] = conn.in_transaction
        return {"ok": True}

    run(
        tool_name="readonly_probe",
        parse_args=_parse_args,
        work=work,
        argv=["--project", project],
    )
    assert seen["in_transaction"] is False


def test_mutating_work_opens_a_transaction(seeded_project):
    """The same probe under ``mutates=True`` sees an open transaction around
    work() — the seam is active."""
    project = seeded_project["project"]
    seen: dict = {}

    def work(*, conn, args, session_id) -> dict:
        seen["in_transaction"] = conn.in_transaction
        return {"ok": True}

    run(
        tool_name="mutating_probe",
        parse_args=_parse_args,
        work=work,
        argv=["--project", project],
        mutates=True,
    )
    assert seen["in_transaction"] is True


def _never(*, conn, args, session_id) -> dict:  # pragma: no cover - never runs
    raise AssertionError("work() must not run when arg parsing fails")


def test_bad_flag_emits_json_envelope_not_usage_dump(capsys):
    """A malformed flag becomes the ``{"error","code"}`` envelope with a
    non-zero exit — argparse's raw usage dump never reaches stderr, so agents
    parse one shape (issue #402)."""
    with pytest.raises(SystemExit) as exc:
        run(
            tool_name="probe",
            parse_args=_parse_args,
            work=_never,
            argv=["--bogus-flag"],
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    # stdout carries the envelope agents parse — not argparse's usage dump,
    # which argparse writes to stderr only.
    out = json.loads(captured.out)  # whole stdout is one JSON object
    assert out["code"] == "USAGE_ERROR"
    assert "error" in out


def test_session_resolution_failure_closes_conn_keeps_envelope(
    seeded_project, capsys, monkeypatch
):
    """If ``open_db`` succeeds but session resolution then raises, the runner
    still closes the conn (no leak) and emits the standard error envelope —
    even though no audit row can be written (issue #407). We wrap ``open_db`` to
    record close calls and force session resolution to raise.
    """
    project = seeded_project["project"]
    closed: list[bool] = []
    real_open_db = run.__globals__["open_db"]

    class TrackingConn:
        """Proxy that delegates to a real apsw conn but records close().

        apsw.Connection.close is read-only, so we can't patch it in place;
        wrapping is the least-magic way to observe that the runner closed it.
        """

        def __init__(self, inner):
            self._inner = inner

        def close(self, *a, **k):
            closed.append(True)
            return self._inner.close(*a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def tracking_open_db(name):
        return TrackingConn(real_open_db(name))

    monkeypatch.setattr("bartleby.skill_runner.open_db", tracking_open_db)

    def boom_session(*a, **k):
        raise RuntimeError("session resolution exploded")

    monkeypatch.setattr("bartleby.skill_runner.ensure_active_session", boom_session)

    with pytest.raises(SystemExit) as exc:
        run(
            tool_name="session_fail_probe",
            parse_args=_parse_args,
            work=_never,
            argv=["--project", project],
        )
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"
    assert "error" in out

    # The conn opened before the failure was closed — no leak.
    assert closed == [True]
    # No session resolved → no audit row for this tool.
    assert _audit_rows(project, "session_fail_probe") == 0


def test_no_active_project_emits_envelope_without_opening_db(capsys, monkeypatch):
    """With no ``--project`` and no active project, the runner raises the
    ``NO_ACTIVE_PROJECT`` SkillError *before* ``open_db`` — so the envelope
    surfaces with exit 1 and the DB is never touched (no conn to leak, no audit
    row). The autouse home-isolation fixture already guarantees a fresh sandbox
    with no active-project pointer; we also assert ``open_db`` is never called.
    """
    def boom_open_db(name):  # pragma: no cover - must never run
        raise AssertionError("open_db must not run when no project resolves")

    monkeypatch.setattr("bartleby.skill_runner.open_db", boom_open_db)

    with pytest.raises(SystemExit) as exc:
        run(
            tool_name="no_project_probe",
            parse_args=_parse_args,
            work=_never,
            argv=[],  # no --project, and the sandbox has no active project
        )
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NO_ACTIVE_PROJECT"
    assert "error" in out


def test_help_still_exits_zero(capsys):
    """``--help`` keeps argparse's clean exit-0 behavior — the envelope arm
    only swallows non-zero usage errors."""
    with pytest.raises(SystemExit) as exc:
        run(
            tool_name="probe",
            parse_args=_parse_args,
            work=_never,
            argv=["--help"],
        )
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out  # argparse printed its own help, untouched
