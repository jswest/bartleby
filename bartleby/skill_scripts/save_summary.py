#!/usr/bin/env python3
"""save_summary — write (or replace) the agent-authored summary for a document.

Output:
    {"summary_id": int, "document_id": int, "chunk_ids": [int, ...]}

Title and description are required so summaries written by the agent show up
in ``list_documents`` the same way ingest-time summaries do.
"""

from __future__ import annotations

import argparse

from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_summary_chunks,
)
from bartleby.ingest.chunk import chunk_markdown_string
from bartleby.ingest.embed import embed_texts
from bartleby.ingest.summarize import normalize_authored_date
from bartleby.skill_runner import SkillError, run


_AUTHOR_MODEL = "agent"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="save_summary")
    p.add_argument("--document", type=int, required=True, dest="document_id")
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--description", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument(
        "--authored-date", type=str, default=None, dest="authored_date",
        help="ISO 8601 YYYY-MM-DD. Silently stored as NULL if malformed.",
    )
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if not args.title or not args.title.strip():
        raise SkillError("EMPTY_TITLE", "Summary title must be non-empty.")
    if not args.description or not args.description.strip():
        raise SkillError(
            "EMPTY_DESCRIPTION", "Summary description must be non-empty."
        )
    if not args.text or not args.text.strip():
        raise SkillError("EMPTY_TEXT", "Summary text must be non-empty.")

    cur = conn.cursor()
    doc_row = cur.execute(
        "SELECT document_id FROM documents WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()
    if doc_row is None:
        raise SkillError(
            "DOCUMENT_NOT_FOUND",
            f"No document with id {args.document_id}.",
        )

    prior = cur.execute(
        "SELECT summary_id FROM summaries WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()
    if prior:
        delete_chunks_for(conn, "summary", prior[0])
        cur.execute("DELETE FROM summaries WHERE summary_id = ?", (prior[0],))

    cur.execute(
        "INSERT INTO summaries "
        "(document_id, title, description, text, model, authored_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (args.document_id, args.title, args.description, args.text,
         _AUTHOR_MODEL, normalize_authored_date(args.authored_date)),
    )
    summary_id = conn.last_insert_rowid()

    rows = chunk_markdown_string(args.text)
    chunk_ids: list[int] = []
    if rows:
        embeddings = embed_texts([r.text for r in rows])
        chunk_inputs = [
            ChunkInput(
                text=row.text,
                embedding=emb,
                chunk_index=i,
                section_heading=row.section_heading,
                content_type=row.content_type,
            )
            for i, (row, emb) in enumerate(zip(rows, embeddings))
        ]
        chunk_ids = insert_summary_chunks(conn, summary_id, chunk_inputs)

    return {
        "summary_id": summary_id,
        "document_id": args.document_id,
        "chunk_ids": chunk_ids,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="save_summary", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
