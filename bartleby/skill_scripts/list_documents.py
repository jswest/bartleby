#!/usr/bin/env python3
"""list_documents — enumerate documents in the corpus.

Three output tiers. The default row is id, file_name, title, description,
authored_date, created_at, has_summary, image_count. ``--verbose`` adds the
full row (page_count, token_count, chunk_count). ``--brief`` drops below the
default to just id, file_name, title — the skinniest useful projection for
triage. ``--verbose`` and ``--brief`` are mutually exclusive.

Output:
    {
      "documents": [{...}, ...],
      "total": int,                  # documents matching all active filters
      "offset": int, "limit": int, "verbose": bool,
      "excluded_null_dated": int,    # NULL-dated docs hidden by a date bound
      "hint": str|null               # set when more pages remain
    }

``title``, ``description``, and ``authored_date`` come from the document's
summary row and are null until one is written (either at ingest time or via
``save_summary``). ``chunk_count`` counts text-track chunks
(``source_kind='document'``); image chunks live under ``source_kind='image'``
and are surfaced via ``image_count``.

Ordering: ``--sort`` picks the order, applied before pagination. ``id``
(default) is ingest order — stable and cheap, the right default for an agent
paging the whole corpus. ``title`` sorts alphabetically by title (falling back
to file_name for unsummarized docs), case-insensitive — the natural order for a
human browsing the list. ``date`` sorts newest-first by ``authored_date`` with
undated documents last. All three break ties on ``document_id`` for determinism.

Date filtering: ``--authored-after`` / ``--authored-before`` bound
``authored_date`` (inclusive, composable with ``--tag``). ``authored_date`` is
summarizer-inferred and stored NULL on anything that isn't a clean
``YYYY-MM-DD`` — often the majority of a corpus. A date bound therefore cannot
be satisfied by an undated document, so those are **excluded by default** and
their count is reported as ``excluded_null_dated`` (so a hidden slice is never
silent). Pass ``--include-nulls`` to keep undated documents in the result
despite an active date bound.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run
from bartleby.skill_scripts._common import add_date_filter_args


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("list_documents", __doc__)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    tier = p.add_mutually_exclusive_group()
    tier.add_argument(
        "--verbose",
        action="store_true",
        help="Include page_count, token_count, chunk_count.",
    )
    tier.add_argument(
        "--brief",
        action="store_true",
        help="Skinniest tier below the default: id, file_name, title only.",
    )
    p.add_argument(
        "--tag",
        action="append", default=None, dest="tags",
        help=(
            "Filter to documents carrying this tag. Repeat for OR semantics "
            "(e.g. --tag ch --tag nyseg). Unknown tag names raise."
        ),
    )
    add_date_filter_args(p)
    p.add_argument(
        "--sort",
        choices=["id", "title", "date"], default="id",
        help=(
            "Result order, applied before pagination. id (default) = ingest "
            "order; title = alphabetical by title/file_name; date = "
            "newest-first by authored_date, undated last."
        ),
    )
    return p.parse_args(argv)


# Maps --sort to an ORDER BY body. Every option ends on document_id so the order
# is total and pagination is stable across pages. Static (no user input reaches
# the SQL), so safe to interpolate.
_ORDER_BY = {
    "id": "d.document_id",
    "title": "COALESCE(s.title, d.file_name) COLLATE NOCASE, d.document_id",
    "date": "(s.authored_date IS NULL), s.authored_date DESC, d.document_id",
}


def work(*, conn, args, session_id) -> dict:
    from bartleby.skill_scripts._tags import resolve_scope

    cur = conn.cursor()
    scope = resolve_scope(
        conn,
        tags=args.tags,
        authored_after=args.authored_after,
        authored_before=args.authored_before,
        include_nulls=args.include_nulls,
    )

    pred, where_params = scope.restrict_in("d.document_id")
    where_clause = f"WHERE {pred} " if pred else ""

    excluded_null_dated = scope.excluded_null_dated

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
        f"ORDER BY {_ORDER_BY[args.sort]} LIMIT ? OFFSET ?",
        [*where_params, args.limit, args.offset],
    )

    documents = []
    for (
        doc_id, file_name, page_count, token_count, created_at,
        title, description, authored_date, has_summary, chunk_count, image_count,
    ) in rows:
        if args.brief:
            documents.append({"id": doc_id, "file_name": file_name, "title": title})
            continue
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
        "excluded_null_dated": excluded_null_dated,
        "hint": hint,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="list_documents", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
