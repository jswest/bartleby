#!/usr/bin/env python3
"""list_documents — enumerate documents in the corpus.

Default output is *brief*: id, file_name, title, description, authored_date,
created_at, has_summary, image_count. Pass ``--verbose`` for the full row
(adds page_count, token_count, chunk_count).

Output:
    {
      "documents": [{...}, ...],
      "total": int,
      "offset": int, "limit": int, "verbose": bool,
      "hint": str|null         # set when more pages remain
    }

``title``, ``description``, and ``authored_date`` come from the document's
summary row and are null until one is written (either at ingest time or via
``save_summary``). ``chunk_count`` counts text-track chunks
(``source_kind='document'``); image chunks live under ``source_kind='image'``
and are surfaced via ``image_count``.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("list_documents", __doc__)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Include page_count, token_count, chunk_count.",
    )
    p.add_argument(
        "--tag",
        action="append", default=None, dest="tags",
        help=(
            "Filter to documents carrying this tag. Repeat for OR semantics "
            "(e.g. --tag ch --tag nyseg). Unknown tag names raise."
        ),
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    from bartleby.skill_scripts._tags import resolve_tag_names

    cur = conn.cursor()
    tag_ids = resolve_tag_names(conn, args.tags) if args.tags else None

    where_clause, where_params = "", []
    if tag_ids is not None:
        tag_ph = ",".join("?" * len(tag_ids))
        where_clause = (
            f"WHERE d.document_id IN ("
            f"  SELECT DISTINCT document_id FROM document_tags "
            f"  WHERE tag_id IN ({tag_ph})"
            f") "
        )
        where_params = list(tag_ids)

    total = cur.execute(
        f"SELECT COUNT(*) FROM documents d {where_clause}",
        where_params,
    ).fetchone()[0]

    rows = cur.execute(
        "SELECT d.document_id, d.file_name, d.page_count, d.token_count, d.created_at, "
        "       s.title AS summary_title, s.description AS summary_description, "
        "       s.authored_date AS summary_authored_date, "
        "       (s.summary_id IS NOT NULL) AS has_summary, "
        "       COALESCE(cc.n, 0) AS chunk_count, "
        "       COALESCE(ic.n, 0) AS image_count "
        "FROM documents d "
        "LEFT JOIN summaries s USING (document_id) "
        "LEFT JOIN (SELECT source_id, COUNT(*) AS n FROM chunks "
        "           WHERE source_kind = 'document' GROUP BY source_id) cc "
        "  ON cc.source_id = d.document_id "
        "LEFT JOIN (SELECT document_id, COUNT(DISTINCT image_id) AS n "
        "           FROM document_images GROUP BY document_id) ic "
        "  ON ic.document_id = d.document_id "
        f"{where_clause}"
        "ORDER BY d.document_id LIMIT ? OFFSET ?",
        [*where_params, args.limit, args.offset],
    )

    documents = []
    for (
        doc_id, file_name, page_count, token_count, created_at,
        title, description, authored_date, has_summary, chunk_count, image_count,
    ) in rows:
        doc = {
            "id": doc_id,
            "file_name": file_name,
            "title": title,
            "description": description,
            "authored_date": authored_date,
            "created_at": created_at,
            "has_summary": bool(has_summary),
            "image_count": image_count,
        }
        if args.verbose:
            doc.update({
                "page_count": page_count,
                "token_count": token_count,
                "chunk_count": chunk_count,
            })
        documents.append(doc)

    next_offset = args.offset + len(documents)
    has_more = next_offset < total
    hint = (
        f"Showing {args.offset + 1}-{next_offset} of {total}. "
        f"Pass --offset {next_offset} to continue."
        if has_more and documents else None
    )

    return {
        "documents": documents,
        "total": total,
        "offset": args.offset,
        "limit": args.limit,
        "verbose": args.verbose,
        "hint": hint,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="list_documents", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
