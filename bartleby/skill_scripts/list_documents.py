#!/usr/bin/env python3
"""list_documents — enumerate documents in the corpus.

Output:
    {
      "documents": [{
        "id": int, "file_name": str, "page_count": int|null,
        "token_count": int|null, "has_summary": bool,
        "chunk_count": int, "created_at": str,
      }, ...],
      "total": int
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="list_documents")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    rows = cur.execute(
        "SELECT d.document_id, d.file_name, d.page_count, d.token_count, d.created_at, "
        "       (s.summary_id IS NOT NULL) AS has_summary, "
        "       COALESCE(cc.n, 0) AS chunk_count "
        "FROM documents d "
        "LEFT JOIN summaries s USING (document_id) "
        "LEFT JOIN (SELECT source_id, COUNT(*) AS n FROM chunks "
        "           WHERE source_kind = 'document' GROUP BY source_id) cc "
        "  ON cc.source_id = d.document_id "
        "ORDER BY d.document_id LIMIT ? OFFSET ?",
        (args.limit, args.offset),
    )

    documents = [
        {
            "id": doc_id,
            "file_name": file_name,
            "page_count": page_count,
            "token_count": token_count,
            "has_summary": bool(has_summary),
            "chunk_count": chunk_count,
            "created_at": created_at,
        }
        for doc_id, file_name, page_count, token_count, created_at,
            has_summary, chunk_count in rows
    ]

    return {"documents": documents, "total": total}


def main(argv: list[str] | None = None) -> None:
    run(tool_name="list_documents", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
