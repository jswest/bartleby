#!/usr/bin/env python3
"""scan — full-text filter over document chunks (FTS5 only, no ranking).

Unlike ``search`` (which fuses full-text + semantic via RRF and returns a
ranked top-N), ``scan`` is a deterministic *filter*: it returns **every**
document chunk that matches a literal full-text query, in document + source
order, paginated with a true ``total`` so you know when you've seen them all.
Use it for corpus-wide enumeration on templated corpora — "for every doc,
give me the chunks containing this marker phrase" — where a literal match is
effectively a filter (e.g. EDGAR/OGE/PACER filings). On heterogeneous corpora
with no uniform marker phrase it degrades to plain ``grep``; reach for
``search`` instead.

Scope: documents only. Summaries, findings, and image chunks are never
returned.

Matching:
    Default is a literal **phrase** — the whole query must appear as a
    contiguous token sequence. Pass ``--match-terms`` to switch to a boolean
    AND of the individual tokens (each must appear somewhere in the chunk, in
    any order).

Ordering (``--sort {document,date}``):
    ``document`` (default) keeps the original ``(document_id, chunk_index)``
    order — document grouping with positional within-doc order, which is what
    makes the ``total``-based termination honest. ``date`` is a document-level
    reorder: walk the matches **oldest-first** by ``authored_date`` (the
    chronological survey of a templated corpus), keeping chunks positional
    within each document. ``authored_date`` is summarizer-inferred and often
    NULL; undated documents sort **last** with ``(document_id, chunk_index)`` as
    the deterministic tiebreaker so pagination stays stable. (Note this is the
    inverse of ``list_documents --sort date``, which is newest-first — scan's
    job is to walk forward in time.) ``--sort`` applies to ``--count-by
    document`` too: ``date`` orders the histogram chronologically instead of by
    hit count.

Output (compact by default):
    Each match carries a snippet truncated to ``--preview`` chars (default
    240; ``…`` appended when trimmed) plus ``text_length`` (pre-truncation).
    To read full bodies, take the ``chunk_id``s and call
    ``read_chunks --chunks <ids>``.

    {
      "query": str,
      "match_mode": "phrase" | "terms",
      "in_documents": [int, ...] | null,
      "tags": [str, ...] | null,
      "offset": int, "limit": int, "total": int,
      "preview": int,
      "matches": [{
        "document_id": int,
        "file_name": str,
        "chunk_id": int,
        "chunk_index": int,
        "page_number": int | null,
        "section_heading": str | null,
        "content_type": str | null,
        "text": str,          # snippet, truncated to `preview`
        "text_length": int,   # pre-truncation length
      }, ...]
    }

With ``--brief`` each match keeps only locators — ``document_id``,
``file_name``, ``chunk_id``, ``page_number`` — dropping ``chunk_index``,
``section_heading``, ``content_type``, ``text``, and ``text_length`` (so
``--preview`` has no effect). Pairs with the always-present ``total`` for pure
"where does this phrase occur" enumeration. The envelope is unchanged.

Count-by aggregate (``--count-by document``):
    Replaces the per-chunk ``matches`` with a per-document hit histogram.
    ``distinct_document_count`` is the headline ("14 documents matched"),
    ``total_chunk_count`` is the old ``total`` ("across 17 chunks"); both are
    the full unpaginated totals. ``--limit``/``--offset`` paginate the
    ``documents`` list (by document, not chunk). Cannot be combined with
    ``--preview`` or ``--brief`` (they only shape per-chunk matches).

    {
      "query": str, "match_mode": "phrase" | "terms",
      "count_by": "document",
      "offset": int, "limit": int,
      "distinct_document_count": int,   # the headline
      "total_chunk_count": int,         # the old `total`
      "documents": [
        {"document_id": int, "file_name": str, "chunk_count": int}, ...
      ]              # default --sort document: chunk_count DESC, document_id;
                     # --sort date: authored_date oldest-first, undated last
    }

Scope (both modes): ``--in-documents`` and ``--tag`` (repeatable, OR
semantics) scope the match the same way they do in ``search``; combined they
intersect. ``--authored-after`` / ``--authored-before`` add inclusive
``YYYY-MM-DD`` date bounds — because ``authored_date`` is summarizer-inferred
and often NULL, a bound excludes undated documents by default (``--include-nulls``
keeps them). Whenever any scope filter is active the response carries a
``filters`` object echoing it — ``{tags, in_documents, authored_after,
authored_before, include_nulls, excluded_null_dated}`` — so the counts are
self-describing; it is absent on an unfiltered scan.
"""

from __future__ import annotations

import argparse
import re

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    add_date_filter_args, apply_preview, comma_int_list, nonneg_int, positive_int,
)
from bartleby.skill_scripts._tags import resolve_scope


DEFAULT_PREVIEW = 240
DEFAULT_LIMIT = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("scan", __doc__)
    p.add_argument("query", type=str)
    p.add_argument(
        "--match-terms",
        action="store_true",
        dest="match_terms",
        help="AND the query's tokens (any order) instead of matching one phrase.",
    )
    p.add_argument(
        "--in-documents",
        type=comma_int_list("document_id"),
        default=None,
        dest="in_documents",
        help="Comma-separated document_ids to restrict the scan to.",
    )
    p.add_argument(
        "--tag", action="append", default=None, dest="tags",
        help="Restrict to documents carrying this tag. Repeat for OR semantics.",
    )
    p.add_argument(
        "--count-by",
        choices=["document"],
        default=None,
        dest="count_by",
        help="Aggregate mode. Instead of per-chunk matches, return a "
             "per-document hit histogram plus distinct_document_count and "
             "total_chunk_count. Cannot be combined with --preview/--brief.",
    )
    p.add_argument(
        "--preview",
        type=positive_int,
        default=None,
        help=f"Truncate each match's text to the first N chars (default {DEFAULT_PREVIEW}).",
    )
    p.add_argument(
        "--brief",
        action="store_true",
        help="Locators only (document_id, file_name, chunk_id, page_number); "
             "drops the text snippet and text_length. Ignores --preview.",
    )
    p.add_argument(
        "--sort",
        choices=["document", "date"], default="document",
        help="Result order. document (default) = (document_id, chunk_index); "
             "date = oldest-first by authored_date (document-level reorder, "
             "undated last). Applies to --count-by document too.",
    )
    p.add_argument("--offset", type=nonneg_int, default=0)
    p.add_argument("--limit", type=positive_int, default=DEFAULT_LIMIT)
    p.add_argument("--project", type=str, default=None)
    add_date_filter_args(p)
    args = p.parse_args(argv)
    if args.count_by and (args.preview is not None or args.brief):
        p.error(
            "--count-by returns a per-document histogram, not per-chunk matches, "
            "so it cannot be combined with --preview or --brief."
        )
    return args


# --sort → ORDER BY body. `date` is a document-level reorder (oldest-first,
# undated last) that keeps chunks positional within each doc; both options end on
# (source_id, chunk_index) so the order is total and pagination is stable. Static
# (no user input reaches the SQL), so safe to interpolate.
_CHUNK_ORDER_BY = {
    "document": "c.source_id, c.chunk_index",
    "date": "(s.authored_date IS NULL), s.authored_date, c.source_id, c.chunk_index",
}
# count-by histogram order. `document` keeps the hit-count ranking; `date`
# reorders the per-document rollup chronologically (undated last).
_DOC_ORDER_BY = {
    "document": "n DESC, c.source_id",
    "date": "(s.authored_date IS NULL), s.authored_date, c.source_id",
}


def _build_fts_query(query: str, match_mode: str) -> str:
    """Build the FTS5 MATCH expression. Returns '' when no tokens remain."""
    if match_mode == "phrase":
        # One quoted phrase. Strip embedded quotes so the string can't break
        # out of the phrase; the tokenizer drops the rest of the punctuation.
        cleaned = re.sub(r"\s+", " ", query.replace('"', " ")).strip()
        return f'"{cleaned}"' if cleaned else ""
    # terms: AND of individually-quoted tokens (FTS5 implicit-AND).
    pieces = [
        f'"{clean}"'
        for token in re.findall(r"\S+", query)
        if (clean := token.replace('"', ""))
    ]
    return " ".join(pieces)


def work(*, conn, args, session_id) -> dict:
    if not args.query or not args.query.strip():
        raise SkillError("EMPTY_QUERY", "Query must be non-empty.")

    match_mode = "terms" if args.match_terms else "phrase"
    scope = resolve_scope(
        conn,
        in_documents=args.in_documents,
        tags=args.tags,
        authored_after=args.authored_after,
        authored_before=args.authored_before,
        include_nulls=args.include_nulls,
    )
    restrict = scope.document_ids  # None = whole corpus, [] = empty, else a set

    def _envelope(extra: dict) -> dict:
        return scope.echo_into({"query": args.query, "match_mode": match_mode, **extra})

    fts_query = _build_fts_query(args.query, match_mode)
    # Nothing can match: an empty token set, or a scope that resolved to no
    # documents (an empty tag / in-documents / date slice).
    no_match = not fts_query or (restrict is not None and not restrict)

    where = "chunks_fts MATCH ? AND c.source_kind = 'document'"
    params: list = [fts_query]
    if restrict is not None:
        placeholders = ",".join("?" * len(restrict))
        where += f" AND c.source_id IN ({placeholders})"
        params.extend(restrict)

    cur = conn.cursor()

    if args.count_by == "document":
        if no_match:
            distinct_document_count = total_chunk_count = 0
            documents = []
        else:
            distinct_document_count, total_chunk_count = cur.execute(
                f"SELECT COUNT(DISTINCT c.source_id), COUNT(*) "
                f"FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
                f"WHERE {where}",
                params,
            ).fetchone()
            documents = [
                {"document_id": source_id, "file_name": file_name,
                 "chunk_count": chunk_count}
                for source_id, file_name, chunk_count in cur.execute(
                    f"SELECT c.source_id, d.file_name, COUNT(*) AS n "
                    f"FROM chunks_fts "
                    f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
                    f"JOIN documents d ON d.document_id = c.source_id "
                    f"LEFT JOIN summaries s ON s.document_id = c.source_id "
                    f"WHERE {where} "
                    f"GROUP BY c.source_id "
                    f"ORDER BY {_DOC_ORDER_BY[args.sort]} "
                    f"LIMIT ? OFFSET ?",
                    [*params, args.limit, args.offset],
                )
            ]
        return _envelope({
            "count_by": "document",
            "offset": args.offset,
            "limit": args.limit,
            "distinct_document_count": distinct_document_count,
            "total_chunk_count": total_chunk_count,
            "documents": documents,
        })

    # Default mode: paginated per-chunk matches.
    preview = args.preview if args.preview is not None else DEFAULT_PREVIEW
    if no_match:
        return _envelope({
            "offset": args.offset, "limit": args.limit, "total": 0,
            "preview": preview, "matches": [],
        })

    total = cur.execute(
        f"SELECT COUNT(*) FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"WHERE {where}",
        params,
    ).fetchone()[0]

    rows = cur.execute(
        f"SELECT c.chunk_id, c.source_id, c.chunk_index, c.section_heading, "
        f"       c.page_number, c.content_type, c.text, d.file_name "
        f"FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"JOIN documents d ON d.document_id = c.source_id "
        f"LEFT JOIN summaries s ON s.document_id = c.source_id "
        f"WHERE {where} "
        f"ORDER BY {_CHUNK_ORDER_BY[args.sort]} "
        f"LIMIT ? OFFSET ?",
        [*params, args.limit, args.offset],
    )

    if args.brief:
        matches = [
            {
                "document_id": source_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "page_number": page_number,
            }
            for (chunk_id, source_id, chunk_index, section_heading,
                 page_number, content_type, text, file_name) in rows
        ]
    else:
        matches = [
            {
                "document_id": source_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "section_heading": section_heading,
                "content_type": content_type,
                "text": apply_preview(text, preview),
                "text_length": len(text),
            }
            for (chunk_id, source_id, chunk_index, section_heading,
                 page_number, content_type, text, file_name) in rows
        ]
    return _envelope({
        "offset": args.offset, "limit": args.limit, "total": total,
        "preview": preview, "matches": matches,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="scan", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
