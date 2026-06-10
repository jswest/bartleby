#!/usr/bin/env python3
"""read_chunks — read chunks by document (paginated), by chunk_id list, or
around a target chunk.

Three modes (mutually exclusive):

  read_chunks --document <id> [--offset N] [--limit N]
      Paginated read of a single document's chunks in chunk_index order.
      Output includes a ``document`` field and pagination metadata.

  read_chunks --chunks 4192,4193,4194
      Direct lookup by chunk_id. Returns those chunks regardless of source.
      Each chunk carries its source_kind/source_id/chunk_index so the agent
      can locate it. Output includes a ``requested`` and ``missing`` list.

  read_chunks --around-chunk <id> [--window N]
      Neighborhood read: returns the target chunk plus N chunks on each
      side (default ``--window 3``), in chunk_index order. Source is
      derived from the target chunk — no need to pass --document. Works
      for any source kind, though image chunks have no neighbors.

In a memory-off session the finding wall (see ``read_finding``) extends here:
finding-kind chunks authored by *another* session are walled off so an
evaluation run can't read prior conclusions by chunk_id. ``--chunks`` drops
such ids into ``missing`` (no text or source_name is returned); an
``--around-chunk`` whose target is a foreign finding chunk raises
``{"code": "MEMORY_OFF"}``. A session's own findings and all
document/summary/image chunks are unaffected.

All modes accept ``--preview N`` to truncate each chunk's ``text`` to the
first ``N`` characters (followed by ``…`` when truncation occurred). Useful
for structural scans when you don't need full prose. Omit ``--preview`` to
get full text. Every returned chunk always carries ``text_length`` — the
pre-truncation length of the chunk's text — so the agent can size-budget
and tell which chunks were trimmed.

Paginated output:
    {
      "mode": "document",
      "document": {"id": int, "file_name": str},
      "offset": int, "limit": int, "total": int,
      "preview": int|null,
      "chunks": [{
        "chunk_id": int, "chunk_index": int,
        "section_heading": str|null,
        "page_number": int|null,
        "content_type": str|null,
        "text": str,
        "text_length": int,
      }, ...]
    }

Direct-lookup output:
    {
      "mode": "chunks",
      "requested": [int, ...],
      "missing": [int, ...],
      "preview": int|null,
      "chunks": [{
        "chunk_id": int,
        "source_kind": str, "source_id": int, "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
        "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
        "text_length": int,
      }, ...]
    }

Around-chunk output:
    {
      "mode": "around",
      "target": {"chunk_id": int, "chunk_index": int,
                 "source_kind": str, "source_id": int, "source_name": str},
      "window": int,
      "preview": int|null,
      "chunks": [{
        "chunk_id": int, "chunk_index": int,
        "section_heading": str|null,
        "page_number": int|null,
        "content_type": str|null,
        "text": str,
        "text_length": int,
      }, ...]
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    apply_preview, chunk_locations, comma_int_list, memory_enabled, nonneg_int,
    positive_int, source_names,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_chunks", __doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--document", type=positive_int, dest="document_id")
    mode.add_argument(
        "--chunks",
        type=comma_int_list("chunk_id"),
        dest="chunk_ids",
        help="Comma-separated chunk_ids to fetch directly.",
    )
    mode.add_argument(
        "--around-chunk",
        type=positive_int,
        dest="around_chunk",
        help="Target chunk_id; returns target plus --window chunks on each side.",
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
        "--window",
        type=nonneg_int,
        default=3,
        help="Used with --around-chunk: neighbors on each side (default 3).",
    )
    p.add_argument(
        "--preview",
        type=positive_int,
        default=None,
        help="Truncate each chunk's text to the first N chars (append '…' if trimmed).",
    )
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def _owned_finding_ids(conn, finding_ids, session_id: int) -> set[int]:
    """Subset of ``finding_ids`` authored by ``session_id``.

    Callers pass the ``source_id`` of finding-kind chunks, which for findings
    equals ``findings.finding_id``.
    """
    ids = list(finding_ids)
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    return {
        row[0]
        for row in conn.cursor().execute(
            f"SELECT finding_id FROM findings "
            f"WHERE finding_id IN ({placeholders}) AND session_id = ?",
            (*ids, session_id),
        )
    }


def _chunks_from_rows(rows, preview: int | None) -> list[dict]:
    return [
        {
            "chunk_id": cid,
            "chunk_index": idx,
            "section_heading": heading,
            "page_number": page,
            "content_type": ctype,
            "text": apply_preview(text, preview),
            "text_length": len(text),
        }
        for cid, idx, heading, page, ctype, text in rows
    ]


def _read_by_chunk_ids(
    conn, chunk_ids: list[int], preview: int | None, *, mem: bool, session_id: int
) -> dict:
    # De-dupe while preserving the agent's requested order.
    seen: set[int] = set()
    ordered: list[int] = []
    for cid in chunk_ids:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    placeholders = ",".join("?" * len(ordered))
    rows = {
        row[0]: row
        for row in conn.cursor().execute(
            f"SELECT chunk_id, source_kind, source_id, chunk_index, "
            f"       section_heading, content_type, text "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            ordered,
        )
    }

    # Memory wall: a memory-off session never sees another session's finding
    # chunks. Drop them before any text/source_name is read, so they fall into
    # ``missing`` as if they didn't exist (mirrors read_finding's MEMORY_OFF).
    if not mem:
        finding_sids = {r[2] for r in rows.values() if r[1] == "finding"}
        owned = _owned_finding_ids(conn, finding_sids, session_id)
        rows = {
            cid: r
            for cid, r in rows.items()
            if not (r[1] == "finding" and r[2] not in owned)
        }

    names = source_names(conn, {(r[1], r[2]) for r in rows.values()})
    locations = chunk_locations(conn, list(rows.keys()))

    missing = [cid for cid in ordered if cid not in rows]
    chunks = []
    for cid in ordered:
        if cid not in rows:
            continue
        _, sk, sid, chunk_index, section_heading, content_type, text = rows[cid]
        loc = locations.get(cid, {"file_name": None, "page_number": None})
        chunks.append({
            "chunk_id": cid,
            "source_kind": sk,
            "source_id": sid,
            "source_name": names.get((sk, sid), ""),
            "file_name": loc["file_name"],
            "page_number": loc["page_number"],
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "content_type": content_type,
            "text": apply_preview(text, preview),
            "text_length": len(text),
        })

    return {
        "mode": "chunks",
        "requested": ordered,
        "missing": missing,
        "preview": preview,
        "chunks": chunks,
    }


def _read_by_document(conn, args) -> dict:
    cur = conn.cursor()
    doc_row = cur.execute(
        "SELECT document_id, file_name FROM documents WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()
    if doc_row is None:
        raise SkillError(
            "DOCUMENT_NOT_FOUND",
            f"No document with id {args.document_id}.",
        )

    total = cur.execute(
        "SELECT COUNT(*) FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ?",
        (args.document_id,),
    ).fetchone()[0]

    rows = list(cur.execute(
        "SELECT chunk_id, chunk_index, section_heading, page_number, "
        "       content_type, text "
        "FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ? "
        "ORDER BY chunk_index LIMIT ? OFFSET ?",
        (args.document_id, args.limit, args.offset),
    ))

    return {
        "mode": "document",
        "document": {"id": doc_row[0], "file_name": doc_row[1]},
        "offset": args.offset,
        "limit": args.limit,
        "total": total,
        "preview": args.preview,
        "chunks": _chunks_from_rows(rows, args.preview),
    }


def _read_around_chunk(conn, args, *, mem: bool, session_id: int) -> dict:
    cur = conn.cursor()
    target = cur.execute(
        "SELECT chunk_id, source_kind, source_id, chunk_index "
        "FROM chunks WHERE chunk_id = ?",
        (args.around_chunk,),
    ).fetchone()
    if target is None:
        raise SkillError(
            "CHUNK_NOT_FOUND",
            f"No chunk with id {args.around_chunk}.",
        )
    target_id, sk, sid, target_idx = target

    # Memory wall: refuse a window centred on another session's finding chunk.
    # The window never crosses (source_kind, source_id), so gating the target
    # also keeps foreign finding chunks out of the neighbourhood.
    if not mem and sk == "finding" and not _owned_finding_ids(conn, {sid}, session_id):
        raise SkillError(
            "MEMORY_OFF",
            f"This session has memory disabled and chunk {args.around_chunk} "
            "belongs to a finding authored by another session, so it is not "
            "accessible. Start a memory-enabled session (omit --no-memory) to "
            "read other sessions' findings.",
        )

    rows = list(cur.execute(
        "SELECT chunk_id, chunk_index, section_heading, page_number, "
        "       content_type, text "
        "FROM chunks "
        "WHERE source_kind = ? AND source_id = ? "
        "  AND chunk_index BETWEEN ? AND ? "
        "ORDER BY chunk_index",
        (sk, sid, target_idx - args.window, target_idx + args.window),
    ))

    name = source_names(conn, {(sk, sid)}).get((sk, sid), "")
    return {
        "mode": "around",
        "target": {
            "chunk_id": target_id,
            "chunk_index": target_idx,
            "source_kind": sk,
            "source_id": sid,
            "source_name": name,
        },
        "window": args.window,
        "preview": args.preview,
        "chunks": _chunks_from_rows(rows, args.preview),
    }


def work(*, conn, args, session_id) -> dict:
    mem = memory_enabled(conn, session_id)
    if args.chunk_ids is not None:
        return _read_by_chunk_ids(
            conn, args.chunk_ids, args.preview, mem=mem, session_id=session_id,
        )
    if args.around_chunk is not None:
        return _read_around_chunk(conn, args, mem=mem, session_id=session_id)
    return _read_by_document(conn, args)


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_chunks", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
