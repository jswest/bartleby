"""Smoke test for skill/scripts/read_chunks.py."""

from __future__ import annotations

import json

import pytest

from bartleby.db.chunks import ChunkInput, insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.session import start_session
from bartleby.skill_scripts import read_chunks
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def _seed_finding_chunk(conn, *, session_id: int, body: str = "finding body") -> int:
    """Insert a finding owned by ``session_id`` and return its body chunk_id."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, "secret finding", "hook", body),
    )
    finding_id = conn.last_insert_rowid()
    emb = [0.01 * i for i in range(EMBEDDING_DIM)]
    [chunk_id] = insert_finding_chunks(
        conn, finding_id,
        [ChunkInput(text=body, embedding=emb, chunk_index=0)],
    )
    return chunk_id


def _other_session(conn, name: str = "author") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)", (name, 1),
    )
    return conn.last_insert_rowid()


def test_read_chunks_happy_path(seeded_project, capsys):
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "document"
    assert out["document"]["file_name"] == "alpha.pdf"
    assert out["total"] == 4
    assert len(out["chunks"]) == 4
    indexes = [c["chunk_index"] for c in out["chunks"]]
    assert indexes == [0, 1, 2, 3]
    assert out["chunks"][0]["section_heading"] == "Intro"


def test_read_chunks_pagination(seeded_project, capsys):
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--offset", "1", "--limit", "2",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 4
    assert [c["chunk_index"] for c in out["chunks"]] == [1, 2]


def test_read_chunks_unknown_document(seeded_project, capsys):
    """An id that's unknown in *both* namespaces: structured error, no hint."""
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", "999",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_DOCUMENT"
    assert "hint" not in out


def _max_chunk_id(project: str) -> int:
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT MAX(chunk_id) FROM chunks"
        ).fetchone()[0]
    finally:
        conn.close()


def _document_id_only(project: str) -> int:
    """Return a document_id that is *not* also a live chunk_id.

    Document and chunk ids share an integer space but autoincrement
    independently, so freshly-inserted (chunkless) documents collide with
    existing chunk_ids until the document counter climbs past max(chunk_id).
    Insert until that holds, so the id is unambiguously document-only.
    """
    conn = open_db(project)
    try:
        cur = conn.cursor()
        max_chunk = cur.execute("SELECT MAX(chunk_id) FROM chunks").fetchone()[0]
        doc_id = 0
        i = 0
        while doc_id <= max_chunk:
            cur.execute(
                "INSERT INTO documents (file_hash, file_name, file_path, "
                "page_count, token_count) VALUES (?, ?, ?, ?, ?)",
                (f"hg{i}", f"gamma{i}.pdf", f"/tmp/gamma{i}.pdf", 1, 10),
            )
            doc_id = conn.last_insert_rowid()
            i += 1
        return doc_id
    finally:
        conn.close()


def test_read_chunks_unknown_document_that_is_a_chunk_id(seeded_project, capsys):
    """--document with a live chunk_id errors with a 'did you mean --chunks' hint.

    Use the highest chunk_id, which lies above max(document_id) here, so it is a
    chunk_id but not a document_id.
    """
    chunk_id = _max_chunk_id(seeded_project["project"])
    assert chunk_id not in {seeded_project["doc_a"], seeded_project["doc_b"]}
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", str(chunk_id),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_DOCUMENT"
    assert out["hint"] == (
        f"{chunk_id} is a chunk_id — did you mean --chunks {chunk_id}?"
    )


def test_read_chunks_by_id_hints_when_missing_is_a_document_id(seeded_project, capsys):
    """A --chunks miss that is a live document_id gets a 'did you mean --document' hint."""
    # A document-only id (no matching chunk_id) misses the chunk lookup, so the
    # cross-namespace hint fires.
    doc_id = _document_id_only(seeded_project["project"])

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", str(doc_id),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == [doc_id]
    assert out["chunks"] == []
    assert out["hints"] == {
        str(doc_id): f"{doc_id} is a document_id — did you mean --document {doc_id}?"
    }


def test_read_chunks_by_id_no_hint_for_valid_id(seeded_project, capsys):
    """A valid chunk_id lookup carries no hints field — silent on success."""
    chunk_id = _doc_chunk_ids(seeded_project["project"], seeded_project["doc_a"])[0]
    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", str(chunk_id),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == []
    assert [c["chunk_id"] for c in out["chunks"]] == [chunk_id]
    assert "hints" not in out


def _doc_chunk_ids(project: str, document_id: int) -> list[int]:
    conn = open_db(project)
    try:
        rows = conn.cursor().execute(
            "SELECT chunk_id FROM chunks "
            "WHERE source_kind = 'document' AND source_id = ? "
            "ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def test_read_chunks_document_mode_includes_page_number(seeded_project, capsys):
    """page_number is parsed from a 'page N' section_heading and is null otherwise."""
    from bartleby.db.connection import open_db
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM
    conn = open_db(seeded_project["project"])
    try:
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_document_chunks(conn, seeded_project["doc_a"], [
            ChunkInput(
                text="alpha page 5 body",
                embedding=emb, chunk_index=4,
                section_heading=None, page_number=5, content_type="text",
            ),
        ])
    finally:
        conn.close()

    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    pages = [c["page_number"] for c in out["chunks"]]
    # The four seeded chunks have no page_number; the new one is page 5.
    assert pages == [None, None, None, None, 5]


def test_read_chunks_by_id_returns_requested(seeded_project, capsys):
    chunk_ids = _doc_chunk_ids(seeded_project["project"], seeded_project["doc_a"])
    target = [chunk_ids[2], chunk_ids[0]]  # arbitrary order

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", ",".join(str(c) for c in target),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "chunks"
    assert out["requested"] == target
    assert out["missing"] == []
    # Returned in requested order, with source metadata populated.
    assert [c["chunk_id"] for c in out["chunks"]] == target
    for c in out["chunks"]:
        assert c["source_kind"] == "document"
        assert c["source_id"] == seeded_project["doc_a"]
        assert c["source_name"] == "alpha.pdf"
        assert c["file_name"] == "alpha.pdf"
        assert c["page_number"] is None    # seeded headings aren't 'page N'
        assert "chunk_index" in c


def test_read_chunks_by_id_reports_missing(seeded_project, capsys):
    chunk_ids = _doc_chunk_ids(seeded_project["project"], seeded_project["doc_a"])
    target = [chunk_ids[0], 999999]

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", ",".join(str(c) for c in target),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == [999999]
    assert [c["chunk_id"] for c in out["chunks"]] == [chunk_ids[0]]


def test_read_chunks_requires_document_or_chunks(seeded_project, capsys):
    with pytest.raises(SystemExit):
        read_chunks.main(["--project", seeded_project["project"]])


def test_read_chunks_document_and_chunks_mutually_exclusive(seeded_project, capsys):
    with pytest.raises(SystemExit):
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
            "--chunks", "1,2,3",
        ])


def test_read_chunks_preview_truncates_text(seeded_project, capsys):
    """--preview N truncates each chunk's text; text_length always reports the original."""
    from bartleby.db.connection import open_db
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM
    conn = open_db(seeded_project["project"])
    try:
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_document_chunks(conn, seeded_project["doc_a"], [
            ChunkInput(
                text="x" * 5000, embedding=emb, chunk_index=10,
                section_heading=None, page_number=None, content_type="text",
            ),
            ChunkInput(
                text="tiny", embedding=emb, chunk_index=11,
                section_heading=None, page_number=None, content_type="text",
            ),
        ])
    finally:
        conn.close()

    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--offset", "4", "--limit", "2",
        "--preview", "50",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["preview"] == 50
    long_chunk, short_chunk = out["chunks"]
    assert long_chunk["text"] == ("x" * 50) + "…"
    assert long_chunk["text_length"] == 5000
    assert short_chunk["text"] == "tiny"
    assert short_chunk["text_length"] == 4


def test_read_chunks_emits_text_length_without_preview(seeded_project, capsys):
    """text_length is always present and reflects the actual stored text length."""
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["preview"] is None
    for chunk in out["chunks"]:
        assert chunk["text_length"] == len(chunk["text"])
        assert not chunk["text"].endswith("…")


def test_read_chunks_preview_in_chunk_id_mode(seeded_project, capsys):
    from bartleby.db.connection import open_db
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM
    conn = open_db(seeded_project["project"])
    try:
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_document_chunks(conn, seeded_project["doc_a"], [
            ChunkInput(
                text="y" * 2000, embedding=emb, chunk_index=20,
                section_heading=None, page_number=None, content_type="text",
            ),
        ])
    finally:
        conn.close()
    chunk_ids = _doc_chunk_ids(seeded_project["project"], seeded_project["doc_a"])
    target = chunk_ids[-1]

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", str(target),
        "--preview", "100",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["preview"] == 100
    chunk = out["chunks"][0]
    assert chunk["text"] == ("y" * 100) + "…"
    assert chunk["text_length"] == 2000


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_read_chunks_preview_rejects_invalid(seeded_project, bad):
    with pytest.raises(SystemExit):
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
            "--preview", bad,
        ])


# --- memory-off finding wall (#271) --------------------------------------


def test_read_chunks_memory_off_drops_foreign_finding_chunk(seeded_project, capsys):
    """A memory-off session sees a foreign finding chunk only as missing — no leak."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        author = _other_session(conn)
        foreign = _seed_finding_chunk(
            conn, session_id=author, body="confidential prior conclusion",
        )
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    read_chunks.main(["--project", project, "--chunks", str(foreign)])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == [foreign]
    assert out["chunks"] == []
    # The body text and finding title must not appear anywhere in the response.
    assert "confidential prior conclusion" not in json.dumps(out)
    assert "secret finding" not in json.dumps(out)


def test_read_chunks_memory_off_keeps_own_and_document_chunks(seeded_project, capsys):
    """Only foreign finding chunks drop; own findings and document chunks remain."""
    project = seeded_project["project"]
    info = start_session(project, memory_enabled=False)
    own_session = info["session_id"]

    conn = open_db(project)
    try:
        author = _other_session(conn)
        foreign = _seed_finding_chunk(conn, session_id=author, body="foreign body")
        own = _seed_finding_chunk(conn, session_id=own_session, body="my own body")
    finally:
        conn.close()
    doc_chunk = _doc_chunk_ids(project, seeded_project["doc_a"])[0]

    read_chunks.main([
        "--project", project,
        "--chunks", f"{foreign},{own},{doc_chunk}",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == [foreign]
    assert [c["chunk_id"] for c in out["chunks"]] == [own, doc_chunk]
    assert "foreign body" not in json.dumps(out)


def test_read_chunks_memory_on_returns_foreign_finding_chunk(seeded_project, capsys):
    """Positive control: a memory-on session reads a foreign finding chunk normally."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        author = _other_session(conn)
        foreign = _seed_finding_chunk(conn, session_id=author, body="visible body")
    finally:
        conn.close()
    # No start_session → the default active session is memory-on.

    read_chunks.main(["--project", project, "--chunks", str(foreign)])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == []
    assert [c["chunk_id"] for c in out["chunks"]] == [foreign]
    assert out["chunks"][0]["text"] == "visible body"
    assert out["chunks"][0]["source_kind"] == "finding"


def test_read_chunks_around_memory_off_foreign_finding_walled(seeded_project, capsys):
    """--around-chunk on a foreign finding chunk raises MEMORY_OFF."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        author = _other_session(conn)
        foreign = _seed_finding_chunk(conn, session_id=author)
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    with pytest.raises(SystemExit) as exc:
        read_chunks.main(["--project", project, "--around-chunk", str(foreign)])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"


def test_read_chunks_around_memory_off_own_finding_allowed(seeded_project, capsys):
    """--around-chunk on the session's own finding chunk still works memory-off."""
    project = seeded_project["project"]
    info = start_session(project, memory_enabled=False)

    conn = open_db(project)
    try:
        own = _seed_finding_chunk(conn, session_id=info["session_id"])
    finally:
        conn.close()

    read_chunks.main(["--project", project, "--around-chunk", str(own)])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "around"
    assert out["target"]["chunk_id"] == own
    assert [c["chunk_id"] for c in out["chunks"]] == [own]
