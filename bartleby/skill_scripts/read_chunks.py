#!/usr/bin/env python3
"""read_chunks — read chunks by document (paginated) or by chunk_id list.

Two modes (mutually exclusive):

  read_chunks --document <id> [--offset N] [--limit N]
      Paginated read of a single document's chunks in chunk_index order.
      Output includes a ``document`` field and pagination metadata.

  read_chunks --chunks 4192,4193,4194
      Direct lookup by chunk_id. Returns those chunks regardless of source.
      Each chunk carries its source_kind/source_id/chunk_index so the agent
      can locate it. Output includes a ``requested`` and ``missing`` list.

Paginated output:
    {
      "mode": "document",
      "document": {"id": int, "file_name": str},
      "offset": int, "limit": int, "total": int,
      "chunks": [{
        "chunk_id": int, "chunk_index": int,
        "section_heading": str|null,
        "page_number": int|null,        # first-class column; null for non-paginated chunks
        "content_type": str|null,
        "text": str,
      }, ...]
    }

Direct-lookup output:
    {
      "mode": "chunks",
      "requested": [int, ...],
      "missing": [int, ...],
      "chunks": [{
        "chunk_id": int,
        "source_kind": str, "source_id": int, "source_name": str,
        "file_name": str|null,          # originating doc (None for findings)
        "page_number": int|null,        # first-class on doc chunks; image join for image chunks
        "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
      }, ...]
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, run
from bartleby.skill_scripts._common import (
    chunk_locations, comma_int_list, source_names,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="read_chunks")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--document", type=int, dest="document_id")
    mode.add_argument(
        "--chunks",
        type=comma_int_list("chunk_id"),
        dest="chunk_ids",
        help="Comma-separated chunk_ids to fetch directly.",
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def _read_by_chunk_ids(conn, chunk_ids: list[int]) -> dict:
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
            "text": text,
        })

    return {
        "mode": "chunks",
        "requested": ordered,
        "missing": missing,
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

    chunks = [
        {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "page_number": page_number,
            "content_type": content_type,
            "text": text,
        }
        for chunk_id, chunk_index, section_heading, page_number,
        content_type, text in rows
    ]

    return {
        "mode": "document",
        "document": {"id": doc_row[0], "file_name": doc_row[1]},
        "offset": args.offset,
        "limit": args.limit,
        "total": total,
        "chunks": chunks,
    }


def work(*, conn, args, session_id) -> dict:
    if args.chunk_ids is not None:
        return _read_by_chunk_ids(conn, args.chunk_ids)
    return _read_by_document(conn, args)


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_chunks", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
