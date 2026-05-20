#!/usr/bin/env python3
"""save_finding — persist a finding (markdown + structural citations).

The body comes from ``--body-file`` (not ``--body``) because findings can be
long markdown and shell-escaping them is a nightmare. The citation list is
comma-separated chunk_ids.

Output:
    {
      "finding_id": int,
      "session_id": int, "session_name": str,
      "chunk_ids": [int, ...],
      "citation_count": int
    }
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bartleby.db.chunks import ChunkInput, insert_finding_chunks
from bartleby.ingest.chunk import chunk_markdown_string
from bartleby.ingest.embed import embed_texts
from bartleby.skill_runner import SkillError, run
from bartleby.skill_scripts._common import comma_int_list


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="save_finding")
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--description", type=str, required=True)
    p.add_argument("--body-file", type=str, required=True, dest="body_file")
    p.add_argument(
        "--citations",
        type=comma_int_list("chunk_id"),
        default=None,
        help="Comma-separated chunk_ids the finding rests on.",
    )
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if not args.title or not args.title.strip():
        raise SkillError("EMPTY_TITLE", "Finding title must be non-empty.")
    if not args.description or not args.description.strip():
        raise SkillError(
            "EMPTY_DESCRIPTION", "Finding description must be non-empty."
        )

    body_path = Path(args.body_file)
    if not body_path.exists() or not body_path.is_file():
        raise SkillError(
            "BODY_FILE_NOT_FOUND",
            f"--body-file path does not exist: {body_path}",
        )
    body = body_path.read_text(encoding="utf-8")
    if not body.strip():
        raise SkillError("EMPTY_BODY", "Finding body is empty.")

    citations: list[int] = args.citations or []

    cur = conn.cursor()

    # Verify cited chunk_ids exist; the FK would catch it, but a clear error
    # at this layer is friendlier.
    if citations:
        placeholders = ",".join("?" * len(citations))
        seen = {
            row[0]
            for row in cur.execute(
                f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
                citations,
            )
        }
        missing = sorted(set(citations) - seen)
        if missing:
            raise SkillError(
                "UNKNOWN_CITATIONS",
                f"Unknown chunk_ids: {missing}",
                unknown_chunk_ids=missing,
            )

    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, args.title, args.description, body),
    )
    finding_id = conn.last_insert_rowid()

    rows = chunk_markdown_string(body)
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
        chunk_ids = insert_finding_chunks(conn, finding_id, chunk_inputs)

    if citations:
        cur.executemany(
            "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
            [(finding_id, cid) for cid in citations],
        )

    session_name = cur.execute(
        "SELECT name FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0]

    return {
        "finding_id": finding_id,
        "session_id": session_id,
        "session_name": session_name,
        "chunk_ids": chunk_ids,
        "citation_count": len(citations),
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="save_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
