"""Partial writes leave no trace: a failure mid-write rolls back every table it
touched (ARCHITECTURE.md invariant).
"""

from __future__ import annotations

import argparse
import json

import pytest

from bartleby import skill_runner
from bartleby.db.chunks import ChunkInput
from bartleby.db.chunks import insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.ingest import parsers
from bartleby.ingest.writer import ParsedDocument
from bartleby.ingest.writer import ParsedImage
from bartleby.ingest.writer import Writer
from bartleby.skill_runner import run
from bartleby.skill_scripts import _tags as tags_helpers
from bartleby.skill_scripts import assign_tag
from bartleby.skill_scripts import edit_finding
from bartleby.skill_scripts import merge_tags
from bartleby.skill_scripts import save_finding
from bartleby.skill_scripts import save_summary
from bartleby.skill_scripts import unassign_tag

from tests._skill_fixtures import _emb
from tests._skill_fixtures import assert_chunk_tables_consistent
from tests._skill_fixtures import mock_embed  # noqa: F401
from tests._skill_fixtures import project_env  # noqa: F401
from tests._skill_fixtures import seed_finding_via_main
from tests._skill_fixtures import seeded_project  # noqa: F401
from tests._skill_fixtures import unprefix


# ---------- skill_save_finding ----------


def test_save_finding_failure_mid_write_leaves_no_trace(
    seeded_project, tmp_path, capsys, monkeypatch
):
    """A failure during chunk insertion (after the findings row is inserted)
    rolls back the whole call: no findings row, no finding chunks in any of the
    three chunk tables, no citation rows (issue #340). The runner seam wraps
    ``work()`` in one transaction, so the earlier INSERT INTO findings unwinds
    with the failed chunk write.

    The failure is injected at ``chunks._pack_embedding`` — i.e. *during* the
    chunk insert, after the findings row already landed — so this is a genuine
    mid-write rollback, not a pre-write validation bounce."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cited_id = cur.execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
        findings_before = cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
        finding_chunks_before = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding'"
        ).fetchone()[0]
        citations_before = cur.execute(
            "SELECT COUNT(*) FROM finding_citations"
        ).fetchone()[0]
    finally:
        conn.close()

    def _boom(embedding):
        raise RuntimeError("injected chunk-write failure")

    monkeypatch.setattr("bartleby.db.chunks._pack_embedding", _boom)

    body_file = tmp_path / "f.md"
    body_file.write_text(f"A claim[^chunk:{cited_id}].", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "doomed",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Every affected table is exactly as before — no orphaned finding row,
        # no chunks in any of the three chunk tables, no citations.
        assert cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == \
            findings_before
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding'"
        ).fetchone()[0] == finding_chunks_before
        assert cur.execute(
            "SELECT COUNT(*) FROM finding_citations"
        ).fetchone()[0] == citations_before
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


# ---------- skill_edit_finding ----------


def test_edit_finding_failure_mid_write_leaves_finding_intact(
    seeded_project, tmp_path, capsys, monkeypatch
):
    """A failure during the body re-chunk (after the findings UPDATE) rolls back
    the whole edit: the prior title, body, chunks, and citations all survive
    intact (issue #340). Without the transaction wrap, the UPDATE and the
    ``delete_chunks_for`` inside ``write_finding_chunks`` would have committed
    independently, leaving the new body saved with zero chunks and stale
    citations. The failure is injected at ``chunks._pack_embedding`` — during
    the chunk insert, after the UPDATE — so it's a true mid-write rollback."""
    saved = seed_finding_via_main(
        seeded_project, tmp_path, capsys,
        title="Original title", description="Original description.",
    )
    finding_id = saved["finding_id"]
    fid = unprefix(finding_id)
    a, b = saved["_chunks"]

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        before_title, before_body = cur.execute(
            "SELECT title, body FROM findings WHERE finding_id = ?",
            (fid,),
        ).fetchone()
        before_chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (fid,),
            )
        ]
        before_citations = sorted(
            r[0] for r in cur.execute(
                "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
                (fid,),
            )
        )
    finally:
        conn.close()
    assert before_chunk_texts  # sanity: the finding had body chunks

    def _boom(embedding):
        raise RuntimeError("injected chunk-write failure")

    monkeypatch.setattr("bartleby.db.chunks._pack_embedding", _boom)

    # Edit the body (and title) — the rebuild must fail and roll everything back.
    new_body_file = tmp_path / "doomed.md"
    new_body_file.write_text(f"# Doomed\n\nNew claim[^chunk:{a}].", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", finding_id,
            "--title", "Doomed title",
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Title and body unchanged — the UPDATE rolled back with the failed
        # chunk write.
        title, body = cur.execute(
            "SELECT title, body FROM findings WHERE finding_id = ?",
            (fid,),
        ).fetchone()
        assert title == before_title
        assert body == before_body
        # The original body chunks survive, byte-for-byte.
        chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (fid,),
            )
        ]
        assert chunk_texts == before_chunk_texts
        # And the original citations survive.
        citations = sorted(
            r[0] for r in cur.execute(
                "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
                (fid,),
            )
        )
        assert citations == before_citations
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


# ---------- skill_save_summary ----------


def test_save_summary_failed_replace_preserves_prior_summary_and_chunks(
    seeded_project, capsys, monkeypatch
):
    """A failure mid-replace leaves the prior summary AND its chunks fully
    intact (issue #340). The replace path deletes the prior summary and its
    chunks, inserts the new summary, then inserts the new chunks — under the
    runner's transaction wrap, a failure during that last chunk insert rolls
    the *whole* sequence back, so the prior summary (and its inferred
    authored_date) is not destroyed and ``summaries.document_id`` (UNIQUE) is
    not orphaned.

    The failure is injected at ``chunks._pack_embedding`` — during the new chunk
    insert, *after* the prior summary + chunks were deleted and the new summary
    row inserted — so it is a genuine mid-write rollback, not a pre-write bounce.
    (Embedding itself is hoisted ahead of the first write, so a bad-embedding
    vector would fail before any delete; breaking the chunk insert directly is
    what reaches the post-delete state this test pins.)"""
    doc_a = seeded_project["doc_a"]
    prior_chunk_ids = seeded_project["summary_a_chunk_ids"]
    assert prior_chunk_ids  # the prior summary has chunks to preserve

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        before_summary = cur.execute(
            "SELECT title, description, text FROM summaries WHERE document_id = ?",
            (doc_a,),
        ).fetchone()
        before_chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='summary' "
                "AND chunk_id IN ({}) ORDER BY chunk_index".format(
                    ",".join("?" * len(prior_chunk_ids))
                ),
                tuple(prior_chunk_ids),
            )
        ]
    finally:
        conn.close()
    assert before_summary is not None
    assert len(before_chunk_texts) == len(prior_chunk_ids)

    def _boom(embedding):
        raise RuntimeError("injected chunk-write failure")

    monkeypatch.setattr("bartleby.db.chunks._pack_embedding", _boom)

    with pytest.raises(SystemExit) as exc:
        save_summary.main([
            "--project", seeded_project["project"],
            "--document-id", f"document:{doc_a}",
            "--title", "Doomed v2",
            "--description", "Replacement that must roll back.",
            "--text", "Replacement summary that never lands.",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # The prior summary survives, unchanged — exactly one row, original text.
        rows = cur.execute(
            "SELECT title, description, text FROM summaries WHERE document_id = ?",
            (doc_a,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == before_summary
        # The prior summary's chunks survive in all three tables, byte-for-byte.
        chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT c.text FROM chunks c "
                "JOIN summaries s ON s.summary_id = c.source_id "
                "WHERE c.source_kind='summary' AND s.document_id = ? "
                "ORDER BY c.chunk_index",
                (doc_a,),
            )
        ]
        assert chunk_texts == before_chunk_texts
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


# ---------- skill_tags ----------


def _fail_on_sql(monkeypatch, needle: str) -> None:
    """Make every connection the runner opens raise on the first SQL statement
    whose text contains ``needle`` (issue #340 atomicity injection).

    Installs an apsw exec-trace on the connection ``skill_runner`` opens, so the
    failure fires *inside* ``work()`` — under the runner's transaction wrap —
    rather than before it. A raising exec-trace aborts that one statement and
    propagates, exercising rollback of whatever the same ``work()`` already
    wrote."""
    real_open_db = skill_runner.open_db

    def _patched(project):
        conn = real_open_db(project)

        def _trace(cursor, sql, bindings):
            if needle in sql:
                raise RuntimeError(f"injected failure on: {needle}")
            return True

        conn.set_exec_trace(_trace)
        return conn

    monkeypatch.setattr(skill_runner, "open_db", _patched)


def _seed_tag(project, name="bad_ocr", description="d") -> int:
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO tags (name, description) VALUES (?, ?)",
            (name, description),
        )
        return conn.last_insert_rowid()
    finally:
        conn.close()


def _assignment_count(project, document_id, tag_id) -> int:
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM document_tags "
            "WHERE document_id = ? AND tag_id = ?",
            (document_id, tag_id),
        ).fetchone()[0]
    finally:
        conn.close()


def test_merge_tags_failure_mid_write_leaves_both_tags_untouched(
    seeded_project, capsys, monkeypatch
):
    """A failure between copying assignments and deleting the source tag rolls
    the whole merge back: no assignments copied onto the destination, and the
    source tag (with its assignments) survives (issue #340). Without the
    transaction wrap the copy would commit and the source tag would linger with
    its assignments already duplicated onto the destination."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
        b_id = conn.last_insert_rowid()
        # 'a' tags doc_a + doc_b; 'b' tags nothing. A clean merge would copy two.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?), (?, ?)",
            (seeded_project["doc_a"], a_id, seeded_project["doc_b"], a_id),
        )
    finally:
        conn.close()

    # Fail the source-tag DELETE — the second write, after the copy INSERT.
    _fail_on_sql(monkeypatch, "DELETE FROM tags")

    with pytest.raises(SystemExit) as exc:
        merge_tags.main([
            "--project", seeded_project["project"], "--from", "a", "--into", "b",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Source tag survives with its two assignments — the copy rolled back.
        assert cur.execute(
            "SELECT COUNT(*) FROM tags WHERE name = 'a'"
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (a_id,)
        ).fetchone()[0] == 2
        # Destination carries nothing — no assignments were copied.
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (b_id,)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_assign_tag_failure_mid_batch_assigns_nothing(
    seeded_project, capsys, monkeypatch
):
    """A failure partway through a multi-document assign rolls back the whole
    batch — neither document ends up tagged (issue #340). The per-document loop
    writes doc_a, then the injected failure fires on doc_b's insert; the runner
    transaction unwinds doc_a too."""
    tag_id = _seed_tag(seeded_project["project"])

    calls = {"n": 0}
    real_assign = tags_helpers.assign

    def _assign(conn, document_id, tag_ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure on second document")
        return real_assign(conn, document_id, tag_ids)

    monkeypatch.setattr("bartleby.skill_scripts.assign_tag.assign", _assign)

    with pytest.raises(SystemExit) as exc:
        assign_tag.main([
            "--project", seeded_project["project"],
            "--documents", f"document:{seeded_project['doc_a']},document:{seeded_project['doc_b']}",
            "--tag", "bad_ocr",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    # Neither document is tagged — the first write rolled back with the second.
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 0
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_b"], tag_id,
    ) == 0


def test_unassign_tag_failure_mid_batch_removes_nothing(
    seeded_project, capsys, monkeypatch
):
    """A failure partway through a multi-document unassign rolls back the whole
    batch — both assignments survive (issue #340)."""
    tag_id = _seed_tag(seeded_project["project"])
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().executemany(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            [(seeded_project["doc_a"], tag_id), (seeded_project["doc_b"], tag_id)],
        )
    finally:
        conn.close()

    calls = {"n": 0}
    real_unassign = tags_helpers.unassign

    def _unassign(conn, document_id, tag_id_):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure on second document")
        return real_unassign(conn, document_id, tag_id_)

    monkeypatch.setattr("bartleby.skill_scripts.unassign_tag.unassign", _unassign)

    with pytest.raises(SystemExit) as exc:
        unassign_tag.main([
            "--project", seeded_project["project"],
            "--documents", f"document:{seeded_project['doc_a']},document:{seeded_project['doc_b']}",
            "--tag", "bad_ocr",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    # Both assignments survive — the first delete rolled back with the second.
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 1
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_b"], tag_id,
    ) == 1


# ---------- skill_runner ----------


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=None)
    return p.parse_args(argv)


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


# ---------- scribe ----------


def test_persist_parse_rolls_back_on_mid_unit_failure(project_env, tmp_path):
    """persist_parse is one transaction: a failure *after* the documents INSERT
    leaves no trace in documents/chunks/images.

    The "atomic parse" invariant — a documents row implies a finished parse, so
    resume only re-runs missing units — rides entirely on persist_parse's single
    ``with self.conn`` (writer.py). This test forces a failure mid-unit (a
    wrong-sized embedding on the second document chunk, which _validate rejects
    after the documents row is already inserted) and asserts the whole unit
    rolled back. A refactor that split that transaction would pass the rest of
    the suite while silently committing a partial parse that reads as complete
    (load-bearing under #254, which writes N+1 documents rows per file).
    """
    good, bad = _emb(), [0.1] * (EMBEDDING_DIM + 1)
    parsed = ParsedDocument(
        file_hash="atomic", file_name="atomic.pdf",
        archive_path=tmp_path / "atomic.pdf", page_count=1, token_count=2,
        document_chunks=[
            ChunkInput(text="First chunk lands fine.", embedding=good, chunk_index=0),
            # Second chunk's embedding is one dim too long: insert_document_chunks
            # validates the batch and raises *after* the documents INSERT.
            ChunkInput(text="Second chunk is poison.", embedding=bad, chunk_index=1),
        ],
        images=[ParsedImage(
            hash="atomic-img", archive_path=tmp_path / "i.jpg",
            width=10, height=10, page_number=1, image_index_on_page=1,
        )],
    )

    conn = open_db(project_env)
    try:
        writer = Writer(conn)
        with pytest.raises(ValueError, match="dims"):
            writer.persist_parse(parsed)

        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
    finally:
        conn.close()


# ---------- ingest_edgar ----------


_ANCHORED_FILING = b"""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<div>
<a href="#sec_business">Business</a>
<a href="#sec_risk">Risk Factors</a>
<a href="#sec_glossary">Glossary of Terms</a>
</div>
<h2 id="sec_business">Business</h2>
<p>We build rockets and launch them into orbit for customers worldwide today.</p>
<p>Our revenue comes from launch services and satellite internet broadband.</p>
<h2 id="sec_risk">Risk Factors</h2>
<p>Rockets are dangerous and may explode on the launch pad without any warning.</p>
<p>Competition in the launch market is intense and is growing larger every year.</p>
<h2 id="sec_glossary">Glossary of Terms</h2>
<p>Apogee is the highest point in an orbit reached by a spacecraft during flight.</p>
</body>
</html>
"""


def _write(tmp_path, name, content: bytes):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_persist_parse_split_is_atomic(project_env, tmp_path):
    """A failure mid-split rolls back the whole unit — no container, no section
    rows, no chunks — so resume (keyed on the container's file_hash) re-parses
    cleanly. The split writes N+1 rows in one transaction (#254/#358)."""
    src = _write(tmp_path, "filing.htm", _ANCHORED_FILING)
    parsed = parsers._parse_html_sec2md(
        src, file_hash="container-hash", file_name="filing.htm",
    )
    # Poison the last section's chunks with an over-long embedding so the chunk
    # insert validation raises after earlier section rows already landed.
    bad = ChunkInput(
        text="poison", embedding=[0.1] * (EMBEDDING_DIM + 1), chunk_index=99,
    )
    parsed.sections[-1].document_chunks.append(bad)

    conn = open_db(project_env)
    try:
        writer = Writer(conn)
        with pytest.raises(ValueError, match="dims"):
            writer.persist_parse(parsed)
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    finally:
        conn.close()
