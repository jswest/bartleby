"""Shared fixtures and helpers for skill-script tests."""

from __future__ import annotations

import json

import pytest

import bartleby.project
from bartleby.db.chunks import (
    ChunkInput,
    insert_document_chunks,
    insert_finding_chunks,
    insert_image_chunks,
    insert_summary_chunks,
)
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.integrity import check_tri_table_sync


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + 0.001 * j for j in range(EMBEDDING_DIM)]


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    """Autouse stub so finding/summary skill tests never reach the real BGE model.

    Patches the names ``skill_scripts._common`` looks up: a deterministic
    ``embed_texts`` and a single-``ChunkRow`` ``chunk_markdown_string`` (the
    latter keeps chunk_count assertions stable). Only active in modules that
    import it.
    """
    monkeypatch.setattr(
        "bartleby.ingest.embed.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.chunk.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def seed_finding(conn, session_id, *, title="A finding", description="hook",
                 body="body", cited_chunk_ids=()) -> tuple[int, list[int]]:
    """Insert a finding (+ one body chunk, + optional citations) directly.

    Returns ``(finding_id, body_chunk_ids)``. The body chunk goes through the
    typed ``insert_finding_chunks`` helper (chunks discipline).
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, title, description, body),
    )
    finding_id = conn.last_insert_rowid()
    body_chunk_ids = insert_finding_chunks(conn, finding_id, [
        ChunkInput(text=body, embedding=_emb(), chunk_index=0),
    ])
    for chunk_id in cited_chunk_ids:
        cur.execute(
            "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
            (finding_id, chunk_id),
        )
    return finding_id, body_chunk_ids


def seed_finding_via_main(seeded_project, tmp_path, capsys, *, title,
                          description, body_suffix: str = "") -> dict:
    """Save a baseline finding by running ``save_finding.main`` end-to-end.

    Unlike :func:`seed_finding` (a direct DB insert), this drives the real
    save_finding script: it cites the first two ``document`` chunks of
    ``doc_a``, writes a body file marking both citations, invokes the script,
    and returns the parsed JSON response with ``_chunks`` attached.
    """
    from bartleby.skill_scripts import save_finding

    conn = open_db(seeded_project["project"])
    try:
        cited = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 2",
            (seeded_project["doc_a"],),
        ).fetchall()
        a, b = (r[0] for r in cited)
    finally:
        conn.close()

    body_file = tmp_path / "seed.md"
    body_file.write_text(
        f"# Seed\n\nClaim one[^{a}]. Claim two[^{b}].{body_suffix}",
        encoding="utf-8",
    )
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", title,
        "--description", description,
        "--body-file", str(body_file),
    ])
    saved = json.loads(capsys.readouterr().out)
    saved["_chunks"] = (a, b)
    return saved


def assert_chunk_tables_consistent(conn) -> None:
    """Assert ``chunks``, ``chunks_fts``, and ``chunks_vec`` agree.

    Thin wrapper over the shippable :func:`bartleby.integrity.check_tri_table_sync`
    (lifted out in #487 so the command and the suite share one implementation).
    The load-bearing detail — the FTS leg uses FTS5's external-content
    ``'integrity-check'`` in the ``rank=1`` form (the one-argument form is a no-op
    for content/index drift in apsw 3.51 / SQLite 3.51), and the vec leg compares
    rowid sets exactly — lives in that module's docstring now.
    """
    result = check_tri_table_sync(conn)
    assert result.passed, result.detail


@pytest.fixture
def project_env():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    bartleby.project.create_project("p")
    yield "p"


@pytest.fixture
def seeded_project(project_env):
    """A project with two documents (one with a summary) and an active session."""
    conn = open_db(project_env)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("h1", "alpha.pdf", "/tmp/alpha.pdf", 5, 1000),
        )
        doc_a = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("h2", "beta.txt", "/tmp/beta.txt", None, 200),
        )
        doc_b = conn.last_insert_rowid()

        # Document A: 4 chunks
        insert_document_chunks(conn, doc_a, [
            ChunkInput(text="alpha chunk zero about pm25", embedding=_emb(0.0),
                       chunk_index=0, section_heading="Intro", content_type="text"),
            ChunkInput(text="alpha chunk one about equity", embedding=_emb(0.1),
                       chunk_index=1, section_heading="Methods", content_type="text"),
            ChunkInput(text="alpha chunk two on results", embedding=_emb(0.2),
                       chunk_index=2, section_heading="Results", content_type="text"),
            ChunkInput(text="alpha chunk three concludes", embedding=_emb(0.3),
                       chunk_index=3, section_heading="Conclusion", content_type="text"),
        ])

        # Document B: 2 chunks
        insert_document_chunks(conn, doc_b, [
            ChunkInput(text="beta chunk zero says hello", embedding=_emb(1.0),
                       chunk_index=0),
            ChunkInput(text="beta chunk one says farewell", embedding=_emb(1.1),
                       chunk_index=1),
        ])

        # Summary for doc A — seeded WITH chunks so a later replace has prior
        # chunks to delete from all three tables (the save_summary replace test
        # asserts they are gone afterward).
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_a, "Alpha", "Test summary of alpha document.",
             "A summary of alpha.", "test"),
        )
        summary_a = conn.last_insert_rowid()
        summary_a_chunk_ids = insert_summary_chunks(conn, summary_a, [
            ChunkInput(text="alpha summary chunk zero", embedding=_emb(3.0),
                       chunk_index=0),
            ChunkInput(text="alpha summary chunk one", embedding=_emb(3.1),
                       chunk_index=1),
        ])
    finally:
        conn.close()

    return {
        "project": project_env,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "summary_a": summary_a,
        "summary_a_chunk_ids": summary_a_chunk_ids,
    }


@pytest.fixture
def dated_corpus(project_env):
    """A corpus whose filenames encode dates, for `scribe backfill-dates` (#536).

    Wave-1 shared fixture: later authored-date sub-issues reuse it. Every
    document's `file_name` is `<key>__YYYY-MM-DD__slug.md` so a single named-
    capture regex (`(?P<date>\\d{4}-\\d{2}-\\d{2})`) matches all of them. The
    documents exercise every backfill branch:

    - ``summary_doc``   — has a *real* summary (model='test') with a NULL date;
                          backfill should UPDATE the date on it.
    - ``stub_doc``      — has NO summary row; backfill should INSERT a stub.
    - ``dated_doc``     — has a real summary that ALREADY carries a date; backfill
                          leaves it (idempotent) unless --overwrite.
    - ``nomatch_doc``   — `file_name` has no date; backfill should not touch it.
    - ``bad_date_doc``  — `file_name` matches the regex but the captured value is
                          an impossible calendar date (2024-13-40); counted as
                          invalid, never written.
    - ``parent_doc`` / ``section_doc`` — a #254 anchor-split pair sharing one
                          `file_name`; both should get the parent's date (the
                          section via a stub).

    Returns the project name and every document id under those keys.
    """
    conn = open_db(project_env)
    try:
        cur = conn.cursor()

        def _doc(file_hash, file_name, *, parent=None):
            cur.execute(
                "INSERT INTO documents "
                "(file_hash, file_name, file_path, page_count, token_count, "
                " parent_document_id) VALUES (?, ?, ?, ?, ?, ?)",
                (file_hash, file_name, f"/corpus/{file_name}", 1, 100, parent),
            )
            doc_id = conn.last_insert_rowid()
            insert_document_chunks(conn, doc_id, [
                ChunkInput(text=f"body of {file_name}", embedding=_emb(),
                           chunk_index=0),
            ])
            return doc_id

        def _summary(doc_id, *, authored_date, model="test"):
            cur.execute(
                "INSERT INTO summaries "
                "(document_id, title, description, text, model, authored_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, "T", "D", "real summary text", model, authored_date),
            )
            return conn.last_insert_rowid()

        summary_doc = _doc("d1", "A001__2021-03-15__alpha.md")
        _summary(summary_doc, authored_date=None)

        stub_doc = _doc("d2", "B002__2022-07-04__beta.md")

        dated_doc = _doc("d3", "C003__2023-01-09__gamma.md")
        _summary(dated_doc, authored_date="2099-12-31")

        nomatch_doc = _doc("d4", "D004__no-date-here__delta.md")

        bad_date_doc = _doc("d5", "E005__2024-13-40__epsilon.md")

        # #254 anchor-split pair: a section row shares the parent's file_name.
        parent_doc = _doc("d6", "F006__2020-05-20__zeta.md")
        section_doc = _doc("d6-sec", "F006__2020-05-20__zeta.md", parent=parent_doc)
    finally:
        conn.close()

    return {
        "project": project_env,
        "summary_doc": summary_doc,
        "stub_doc": stub_doc,
        "dated_doc": dated_doc,
        "nomatch_doc": nomatch_doc,
        "bad_date_doc": bad_date_doc,
        "parent_doc": parent_doc,
        "section_doc": section_doc,
    }


def seed_image(conn, document_id: int, *, file_hash: str, file_path: str,
               description: str = "A test image scene.",
               ocr_text: str = "WELCOME",
               page_number: int | None = 1,
               image_index_on_page: int = 1) -> int:
    """Helper for image-scope tests: insert an image, link it to a doc, chunk it."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO images "
        "(file_hash, file_path, width, height, analysis_json, analysis_model) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_hash, file_path, 800, 600,
         '{"kind":"scene","text":"' + ocr_text + '","description":"' + description + '","notes":""}',
         "fake-vlm"),
    )
    image_id = conn.last_insert_rowid()
    cur.execute(
        "INSERT INTO document_images "
        "(document_id, image_id, page_number, image_index_on_page) "
        "VALUES (?, ?, ?, ?)",
        (document_id, image_id, page_number, image_index_on_page),
    )
    insert_image_chunks(conn, image_id, [
        ChunkInput(text=ocr_text, embedding=_emb(2.0), chunk_index=0,
                   content_type="image_ocr"),
        ChunkInput(text=description, embedding=_emb(2.1), chunk_index=1,
                   content_type="image_description"),
    ])
    return image_id
