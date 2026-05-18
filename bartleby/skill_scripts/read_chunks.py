#!/usr/bin/env python3
"""read_chunks — paginated read of a document's chunks (in chunk_index order).

Output:
    {
      "document": {"id": int, "file_name": str},
      "offset": int, "limit": int, "total": int,
      "chunks": [{
        "chunk_id": int, "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
      }, ...]
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="read_chunks")
    p.add_argument("--document", type=int, required=True, dest="document_id")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
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
        "SELECT chunk_id, chunk_index, section_heading, content_type, text "
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
            "content_type": content_type,
            "text": text,
        }
        for chunk_id, chunk_index, section_heading, content_type, text in rows
    ]

    return {
        "document": {"id": doc_row[0], "file_name": doc_row[1]},
        "offset": args.offset,
        "limit": args.limit,
        "total": total,
        "chunks": chunks,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_chunks", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
