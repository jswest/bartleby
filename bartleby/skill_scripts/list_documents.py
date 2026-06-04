#!/usr/bin/env python3
"""list_documents — enumerate documents in the corpus.

Default output is *brief*: id, file_name, title, description, authored_date,
created_at, has_summary, image_count. Pass ``--verbose`` for the full row
(adds page_count, token_count, chunk_count).

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

from bartleby.skill_runner import SkillError, build_arg_parser, run


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
    p.add_argument(
        "--authored-after",
        type=str, default=None, dest="authored_after",
        help=(
            "Keep documents whose authored_date is on or after this "
            "YYYY-MM-DD. Composable with --tag and --authored-before."
        ),
    )
    p.add_argument(
        "--authored-before",
        type=str, default=None, dest="authored_before",
        help="Keep documents whose authored_date is on or before this YYYY-MM-DD.",
    )
    p.add_argument(
        "--include-nulls",
        action="store_true", dest="include_nulls",
        help=(
            "With a date bound active, keep NULL-dated (undated) documents "
            "instead of excluding them. No effect without a date bound."
        ),
    )
    return p.parse_args(argv)


def _validate_date(flag: str, raw: str | None) -> str | None:
    """Reject a malformed date *bound* loudly rather than silently nulling it.

    ``normalize_authored_date`` returns None on junk, which is right for
    summarizer-inferred storage but a footgun for a user-supplied filter: a
    silently-dropped bound would widen the result without warning. So we reuse
    its validation but raise instead.
    """
    from bartleby.ingest.summarize import normalize_authored_date

    if raw is None:
        return None
    norm = normalize_authored_date(raw)
    if norm is None:
        raise SkillError(
            "INVALID_DATE",
            f"{flag} must be a real calendar date in YYYY-MM-DD form; got {raw!r}.",
        )
    return norm


def work(*, conn, args, session_id) -> dict:
    from bartleby.skill_scripts._tags import resolve_tag_names

    cur = conn.cursor()
    tag_ids = resolve_tag_names(conn, args.tags) if args.tags else None
    authored_after = _validate_date("--authored-after", args.authored_after)
    authored_before = _validate_date("--authored-before", args.authored_before)

    # Tag filter is the "base" predicate; the date bound layers on top of it so
    # we can separately count the undated docs the bound hides (excluded_null_dated).
    base_parts, base_params = [], []
    if tag_ids is not None:
        tag_ph = ",".join("?" * len(tag_ids))
        base_parts.append(
            f"d.document_id IN ("
            f"SELECT DISTINCT document_id FROM document_tags WHERE tag_id IN ({tag_ph}))"
        )
        base_params.extend(tag_ids)

    date_parts, date_params = [], []
    if authored_after is not None:
        date_parts.append("s.authored_date >= ?")
        date_params.append(authored_after)
    if authored_before is not None:
        date_parts.append("s.authored_date <= ?")
        date_params.append(authored_before)
    date_active = bool(date_parts)

    where_parts = list(base_parts)
    where_params = list(base_params)
    if date_active:
        bounds = " AND ".join(date_parts)
        if args.include_nulls:
            # Undated docs ride along despite the bound.
            where_parts.append(f"(s.authored_date IS NULL OR ({bounds}))")
        else:
            where_parts.append(f"(s.authored_date IS NOT NULL AND {bounds})")
        where_params.extend(date_params)

    def _where(parts: list[str]) -> str:
        return ("WHERE " + " AND ".join(parts) + " ") if parts else ""

    where_clause = _where(where_parts)

    # The date predicate lives on summaries, so the COUNT must join it too.
    # summaries is 0-or-1 per document, so the LEFT JOIN never inflates the count.
    total = cur.execute(
        "SELECT COUNT(*) FROM documents d "
        "LEFT JOIN summaries s USING (document_id) "
        f"{where_clause}",
        where_params,
    ).fetchone()[0]

    # How many base-matching docs the date bound silently dropped for being
    # undated. Only meaningful when a bound is active and we excluded them.
    excluded_null_dated = 0
    if date_active and not args.include_nulls:
        excluded_null_dated = cur.execute(
            "SELECT COUNT(*) FROM documents d "
            "LEFT JOIN summaries s USING (document_id) "
            f"{_where([*base_parts, 's.authored_date IS NULL'])}",
            base_params,
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
        "excluded_null_dated": excluded_null_dated,
        "hint": hint,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="list_documents", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
