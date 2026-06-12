"""Typed insert helpers for the polymorphic ``chunks`` table.

All writes to ``chunks`` (and the parallel ``chunks_fts`` / ``chunks_vec``
virtual tables) go through this module. The three public ``insert_*`` helpers
each hardcode their ``source_kind``; that hardcoding is the point. The
``source_kind`` ``CHECK`` constraint in the schema is the belt; these helpers
are the suspenders.

Outside ``bartleby.db``, no module may ``INSERT INTO chunks`` directly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import apsw

from bartleby.db.schema import ALLOWED_SOURCE_KINDS, EMBEDDING_DIM


@dataclass
class ChunkInput:
    text: str
    embedding: list[float]
    chunk_index: int
    section_heading: str | None = None
    page_number: int | None = None
    content_type: str | None = None


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{EMBEDDING_DIM}f", *embedding)


def _validate(chunks: list[ChunkInput]) -> None:
    prev_index: int | None = None
    for c in chunks:
        if not c.text or not c.text.strip():
            raise ValueError(
                f"chunk text must be non-empty (chunk_index={c.chunk_index})"
            )
        if len(c.embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding has {len(c.embedding)} dims, expected {EMBEDDING_DIM} "
                f"(chunk_index={c.chunk_index})"
            )
        if c.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0 (got {c.chunk_index})")
        if prev_index is not None and c.chunk_index <= prev_index:
            raise ValueError(
                f"chunk_index must be strictly increasing "
                f"(got {c.chunk_index} after {prev_index})"
            )
        prev_index = c.chunk_index


def _insert(
    conn: apsw.Connection,
    source_kind: str,
    source_id: int,
    chunks: list[ChunkInput],
    ingest_run_id: int | None = None,
) -> list[int]:
    if not chunks:
        return []
    _validate(chunks)

    inserted_ids: list[int] = []
    with conn:
        cur = conn.cursor()
        for c in chunks:
            cur.execute(
                "INSERT INTO chunks "
                "(source_kind, source_id, chunk_index, text, "
                " section_heading, page_number, content_type, ingest_run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_kind, source_id, c.chunk_index, c.text,
                 c.section_heading, c.page_number, c.content_type,
                 ingest_run_id),
            )
            chunk_id = conn.last_insert_rowid()
            inserted_ids.append(chunk_id)
            cur.execute(
                "INSERT INTO chunks_fts(rowid, text, section_heading) VALUES (?, ?, ?)",
                (chunk_id, c.text, c.section_heading or ""),
            )
            cur.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                (chunk_id, _pack_embedding(c.embedding)),
            )
    return inserted_ids


def insert_document_chunks(
    conn: apsw.Connection,
    document_id: int,
    chunks: list[ChunkInput],
    ingest_run_id: int | None = None,
) -> list[int]:
    return _insert(conn, "document", document_id, chunks, ingest_run_id)


def insert_summary_chunks(
    conn: apsw.Connection,
    summary_id: int,
    chunks: list[ChunkInput],
    ingest_run_id: int | None = None,
) -> list[int]:
    return _insert(conn, "summary", summary_id, chunks, ingest_run_id)


def insert_finding_chunks(
    conn: apsw.Connection,
    finding_id: int,
    chunks: list[ChunkInput],
) -> list[int]:
    # Findings are authored in a research session, not produced by an ingest
    # run, so they carry no ingest_run_id (the column stays NULL).
    return _insert(conn, "finding", finding_id, chunks)


def insert_image_chunks(
    conn: apsw.Connection,
    image_id: int,
    chunks: list[ChunkInput],
    ingest_run_id: int | None = None,
) -> list[int]:
    return _insert(conn, "image", image_id, chunks, ingest_run_id)


def delete_chunks_of_kind(conn: apsw.Connection, source_kind: str) -> list[int]:
    """Delete every chunk of ``source_kind`` from chunks, chunks_fts, chunks_vec.

    The whole-kind counterpart to :func:`delete_chunks_for` (which keys on a
    single source). Used by corpus publish to drop all finding chunks at once.
    Returns the deleted chunk ids so callers can re-point anything that anchored
    at them (e.g. ``document_tags.chunk_id``).
    """
    if source_kind not in ALLOWED_SOURCE_KINDS:
        raise ValueError(
            f"invalid source_kind {source_kind!r}; "
            f"expected one of {ALLOWED_SOURCE_KINDS}"
        )

    with conn:
        cur = conn.cursor()
        ids = [
            row[0]
            for row in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind = ?",
                (source_kind,),
            )
        ]
        for cid in ids:
            cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
            cur.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        cur.execute("DELETE FROM chunks WHERE source_kind = ?", (source_kind,))
    return ids


def rebuild_fts(conn: apsw.Connection) -> None:
    """Rebuild the external-content ``chunks_fts`` index from ``chunks``.

    The FTS5 ``'rebuild'`` command lives here, in the one module sanctioned to
    issue raw writes against the chunks shadow tables (the chokepoint guard
    exempts only this module). Callers that mutate ``chunks`` outside the
    insert helpers — e.g. publish, which bulk-deletes finding chunks — call this
    to bring the index back in sync.
    """
    conn.cursor().execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")


def delete_chunks_for(
    conn: apsw.Connection,
    source_kind: str,
    source_id: int,
) -> int:
    """Delete all chunks for a source from chunks, chunks_fts, and chunks_vec.

    Returns the number of chunks deleted.
    """
    if source_kind not in ALLOWED_SOURCE_KINDS:
        raise ValueError(
            f"invalid source_kind {source_kind!r}; "
            f"expected one of {ALLOWED_SOURCE_KINDS}"
        )

    with conn:
        cur = conn.cursor()
        ids = [
            row[0]
            for row in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind = ? AND source_id = ?",
                (source_kind, source_id),
            )
        ]
        for cid in ids:
            cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
            cur.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        cur.execute(
            "DELETE FROM chunks WHERE source_kind = ? AND source_id = ?",
            (source_kind, source_id),
        )
    return len(ids)
