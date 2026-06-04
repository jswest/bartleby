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

``--in-documents`` and ``--tag`` (repeatable, OR semantics) scope the match
the same way they do in ``search``; combined they intersect.
"""

from __future__ import annotations

import argparse
import re

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    apply_preview, comma_int_list, nonneg_int, positive_int,
)
from bartleby.skill_scripts._tags import intersect_tag_filter


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
        "--preview",
        type=positive_int,
        default=DEFAULT_PREVIEW,
        help=f"Truncate each match's text to the first N chars (default {DEFAULT_PREVIEW}).",
    )
    p.add_argument("--offset", type=nonneg_int, default=0)
    p.add_argument("--limit", type=positive_int, default=DEFAULT_LIMIT)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


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
    in_documents, tag_names = intersect_tag_filter(conn, args.in_documents, args.tags)

    def _response(matches: list, total: int) -> dict:
        return {
            "query": args.query,
            "match_mode": match_mode,
            "in_documents": in_documents,
            "tags": tag_names,
            "offset": args.offset,
            "limit": args.limit,
            "total": total,
            "preview": args.preview,
            "matches": matches,
        }

    fts_query = _build_fts_query(args.query, match_mode)
    # Empty token set, or a tag/in-documents scope that resolved to no
    # documents: nothing can match.
    if not fts_query or (in_documents is not None and not in_documents):
        return _response([], 0)

    where = "chunks_fts MATCH ? AND c.source_kind = 'document'"
    params: list = [fts_query]
    if in_documents is not None:
        placeholders = ",".join("?" * len(in_documents))
        where += f" AND c.source_id IN ({placeholders})"
        params.extend(in_documents)

    cur = conn.cursor()
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
        f"WHERE {where} "
        f"ORDER BY c.source_id, c.chunk_index "
        f"LIMIT ? OFFSET ?",
        [*params, args.limit, args.offset],
    )

    matches = [
        {
            "document_id": source_id,
            "file_name": file_name,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "section_heading": section_heading,
            "content_type": content_type,
            "text": apply_preview(text, args.preview),
            "text_length": len(text),
        }
        for (chunk_id, source_id, chunk_index, section_heading,
             page_number, content_type, text, file_name) in rows
    ]
    return _response(matches, total)


def main(argv: list[str] | None = None) -> None:
    run(tool_name="scan", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
