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
