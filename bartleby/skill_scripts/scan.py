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
        "authored_date": str | null,   # summarizer-inferred; null for undated docs
        "text": str,          # snippet, truncated to `preview`
        "text_length": int,   # pre-truncation length
      }, ...]
    }

With ``--brief`` each match keeps only locators — ``document_id``,
``file_name``, ``chunk_id``, ``page_number``, ``authored_date`` — dropping
``chunk_index``, ``section_heading``, ``content_type``, ``text``, and
``text_length`` (so ``--preview`` has no effect). ``authored_date`` rides along
even here (it's locator-grade), so brief matches stay sortable/triageable by
time without a re-lookup. Pairs with the always-present ``total`` for pure
"where does this phrase occur" enumeration. The envelope is unchanged.

``--returning <field>,...`` projects each match to exactly the named fields, in
the order given, overriding both the default and ``--brief`` (the envelope is
untouched). Selectable fields: ``chunk_id``, ``document_id``, ``file_name``,
``chunk_index``, ``page_number``, ``section_heading``, ``content_type``,
``authored_date``, ``text``, ``text_length``. An unknown field returns an
``UNKNOWN_RETURNING_FIELD`` error naming the valid set.

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

    ``--returning`` works here too, with its own whitelist: ``chunk_id``,
    ``document_id``, ``file_name``, ``chunk_count``. ``chunk_id`` is the lowest
    matching chunk in that document — a citable handle so you don't re-read to
    recover one (not part of the default histogram row). ``--returning`` is
    rejected on ``--count-by '/regex/'`` (``RETURNING_NOT_APPLICABLE``): a
    captured value spans documents, so its buckets carry no citable id.

Count-by capture (``--count-by '/regex/'``):
    On templated corpora the interesting value often sits in a predictable
    labeled field — "the number after ``H.R.``", "the amount after
    ``Income:``". Pass a ``/regex/`` carrying one capture group instead of the
    ``document`` keyword to bucket matches by that captured substring,
    replacing the corpus-wide hand-rolled-regex habit with a primitive:

        scan "bill" --count-by '/H\\.R\\.\\s*(\\d+)/'
        → groups: [{"value": "4346", "count": 7}, {"value": "815", "count": 3}]

    Counting is **per match**, not per chunk — a chunk with two matches adds
    two — for an honest frequency. A capture that didn't participate
    (``group(1)`` is ``None``) is skipped. ``distinct_value_count`` is the
    headline, ``total_match_count`` the full per-match total; both are the full
    unpaginated totals. Buckets sort by count desc then value asc, paginated by
    ``--limit``/``--offset``; ``--sort`` has no effect (a captured value spans
    documents, so chronological order is meaningless). Like ``--count-by
    document`` it cannot combine with ``--preview``/``--brief``. This is a
    regex-capture-plus-fold primitive, deliberately **not** a query engine — no
    joins, no predicates. The pattern is run with a between-chunk wall-clock
    deadline (``COUNT_BY_TIMEOUT`` on overrun) and a match cap; hitting the cap
    sets ``truncated`` to ``true`` with partial counts.

    {
      "query": str, "match_mode": "phrase" | "terms",
      "count_by": "/regex/",            # echoes the pattern as passed
      "offset": int, "limit": int,
      "distinct_value_count": int,      # the headline
      "total_match_count": int,         # full per-match total
      "truncated": bool,                # true if the match cap clipped counts
      "groups": [
        {"value": str, "count": int}, ...   # count DESC, then value ASC
      ]
    }

Scope (both modes): ``--in-documents`` and ``--tag`` (repeatable, OR
semantics) scope the match the same way they do in ``search``; combined they
intersect. ``--file-like <pattern>`` (SQL ``LIKE``, repeatable for OR) keeps
only documents whose ``file_name`` matches a pattern and ANDs with the other
scopes. ``--heading-like <pattern>`` (SQL ``LIKE``, repeatable for OR) is the
*chunk*-level analogue: it keeps only chunks whose ``section_heading`` matches a
pattern (chunks with no heading never match) and ANDs with the rest.
``--authored-after`` / ``--authored-before`` add inclusive
``YYYY-MM-DD`` date bounds — because ``authored_date`` is summarizer-inferred
and often NULL, a bound excludes undated documents by default (``--include-nulls``
keeps them). Whenever any scope filter is active the response carries a
``filters`` object echoing it — ``{tags, in_documents, file_like, heading_like,
authored_after, authored_before, include_nulls, excluded_null_dated}`` (with
``heading_like`` present only when that flag is) — so the counts are
self-describing; it is absent on an unfiltered scan.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    add_date_filter_args, add_file_like_arg, add_returning_arg, apply_preview,
    comma_int_list, nonneg_int, positive_int, project_row,
)
from bartleby.skill_scripts._tags import resolve_scope


DEFAULT_PREVIEW = 240
DEFAULT_LIMIT = 100

# --returning whitelists. chunk_id + document_id lead every set so a citable id
# is always selectable without a re-read round-trip. The per-chunk match set is
# the full default-mode row; --count-by document rows expose a representative
# chunk_id (the lowest matching chunk in that document) alongside the histogram
# columns. --count-by /regex/ buckets span documents, so no id is selectable
# there — --returning is rejected for the regex aggregate.
MATCH_FIELDS = [
    "chunk_id", "document_id", "file_name", "chunk_index", "page_number",
    "section_heading", "content_type", "authored_date", "text", "text_length",
]
COUNT_BY_DOCUMENT_FIELDS = ["chunk_id", "document_id", "file_name", "chunk_count"]

# Runaway guards for a regex --count-by. The pattern is agent-supplied, not
# attacker-supplied, so this is a footgun rail, not an adversarial-ReDoS defence:
# the deadline is checked *between chunks* (a single re.finditer call is C code
# that can't be interrupted mid-match), which catches the realistic "broad
# pattern × big corpus" runaway. Chunks are size-bounded by the chunker, so the
# residual single-chunk-backtracking gap stays improbable. The match cap bounds
# Counter growth; when hit, the result is flagged truncated rather than dropped
# silently.
COUNT_BY_DEADLINE_SECONDS = 5.0
COUNT_BY_MAX_MATCHES = 100_000


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
    add_file_like_arg(p)
    p.add_argument(
        "--heading-like",
        action="append", default=None, dest="heading_like", metavar="PATTERN",
        help=(
            "Keep only chunks whose section_heading matches this SQL LIKE "
            "pattern (%% = any run, _ = one char), e.g. '2023 Q%%'. Repeat for "
            "OR; the group ANDs with --tag / --in-documents / --file-like / "
            "date bounds. Chunks with no heading never match."
        ),
    )
    p.add_argument(
        "--count-by",
        default=None,
        dest="count_by",
        metavar="document|/regex/",
        help="Aggregate mode. 'document' returns a per-document hit histogram "
             "(distinct_document_count + total_chunk_count). A /regex/ with a "
             "capture group instead buckets matches by the captured substring "
             "(distinct_value_count + total_match_count). Cannot be combined "
             "with --preview/--brief.",
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
        help="Locators only (document_id, file_name, chunk_id, page_number, "
             "authored_date); drops the text snippet and text_length. Ignores "
             "--preview.",
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
    add_date_filter_args(p)
    add_returning_arg(p, MATCH_FIELDS)
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


def _parse_count_by(value: str) -> tuple[str, re.Pattern | None]:
    """Classify a --count-by value.

    Returns ('document', None) for the literal keyword, or ('regex', compiled)
    for a /pattern/ that carries at least one capture group. Raises SkillError
    on a malformed value, an uncompilable pattern, or a capture-less regex.
    """
    if value == "document":
        return "document", None
    if len(value) >= 2 and value.startswith("/") and value.endswith("/"):
        pattern = value[1:-1]
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise SkillError(
                "INVALID_COUNT_BY_REGEX",
                f"--count-by regex {value!r} does not compile: {exc}.",
            )
        if compiled.groups < 1:
            raise SkillError(
                "COUNT_BY_NO_CAPTURE",
                f"--count-by regex {value!r} has no capture group; wrap the "
                "value to bucket on in parentheses, e.g. '/H\\.R\\.\\s*(\\d+)/'.",
            )
        return "regex", compiled
    raise SkillError(
        "INVALID_COUNT_BY",
        f"--count-by must be 'document' or a /regex/ with a capture group; "
        f"got {value!r}.",
    )


def _count_by_regex(cur, where: str, params: list, pattern: re.Pattern) -> tuple[Counter, bool]:
    """Tally pattern's first capture group across every matching chunk's text.

    Counts per match (a chunk with two matches contributes two), skipping a
    capture that didn't participate (group(1) is None). Returns the histogram
    and whether the match cap truncated it. Enforces a between-chunk wall-clock
    deadline — see the COUNT_BY_* guard note above for why it's not mid-match.
    """
    counter: Counter = Counter()
    total = 0
    deadline = time.monotonic() + COUNT_BY_DEADLINE_SECONDS
    rows = cur.execute(
        f"SELECT c.text FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"WHERE {where}",
        params,
    )
    for (text,) in rows:
        if time.monotonic() > deadline:
            raise SkillError(
                "COUNT_BY_TIMEOUT",
                f"--count-by regex exceeded {COUNT_BY_DEADLINE_SECONDS:g}s; "
                "narrow the scan scope or simplify the pattern.",
            )
        for match in pattern.finditer(text):
            value = match.group(1)
            if value is None:
                continue
            counter[value] += 1
            total += 1
            if total >= COUNT_BY_MAX_MATCHES:
                return counter, True  # match cap hit → partial counts
    return counter, False


def _project_count_by_document(full: dict, returning: list[str] | None) -> dict:
    """Project a --count-by document row to --returning, or its default.

    The default row keeps the histogram's three columns (chunk_id is selectable
    via --returning but isn't in the default output — it's the citable handle,
    not part of the histogram contract).
    """
    projected = project_row(full, returning, COUNT_BY_DOCUMENT_FIELDS)
    if projected is not None:
        return projected
    return {
        "document_id": full["document_id"],
        "file_name": full["file_name"],
        "chunk_count": full["chunk_count"],
    }


def work(*, conn, args, session_id) -> dict:
    if not args.query or not args.query.strip():
        raise SkillError("EMPTY_QUERY", "Query must be non-empty.")

    match_mode = "terms" if args.match_terms else "phrase"
    count_mode, count_regex = (None, None)
    if args.count_by is not None:
        count_mode, count_regex = _parse_count_by(args.count_by)
    scope = resolve_scope(
        conn,
        in_documents=args.in_documents,
        tags=args.tags,
        file_like=args.file_like,
        authored_after=args.authored_after,
        authored_before=args.authored_before,
        include_nulls=args.include_nulls,
    )
    restrict = scope.document_ids  # None = whole corpus, [] = empty, else a set

    def _envelope(extra: dict) -> dict:
        env = scope.echo_into({"query": args.query, "match_mode": match_mode, **extra})
        if args.heading_like:
            # --heading-like is a chunk-level filter, so it lives outside Scope's
            # document-level echo; fold it into the same filters object (creating
            # one if the scope alone was unfiltered).
            env.setdefault("filters", {})["heading_like"] = args.heading_like
        return env

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
    if args.heading_like:
        # OR the patterns within the group; the group ANDs with the rest. Pushed
        # down parameterized — the agent's pattern never reaches the SQL text.
        like_clause = " OR ".join("c.section_heading LIKE ?" for _ in args.heading_like)
        where += f" AND ({like_clause})"
        params.extend(args.heading_like)

    cur = conn.cursor()

    if count_mode == "regex":
        if args.returning is not None:
            raise SkillError(
                "RETURNING_NOT_APPLICABLE",
                "--returning has no effect on --count-by '/regex/' output: a "
                "captured value spans documents, so its {value, count} buckets "
                "carry no citable id to project. Use --count-by document for a "
                "per-document histogram with selectable chunk_id/document_id.",
            )
        if no_match:
            counter, truncated = Counter(), False
        else:
            counter, truncated = _count_by_regex(cur, where, params, count_regex)
        # Headline + total are full (pre-pagination); buckets sort by count
        # desc then value asc. --sort doesn't apply: a captured value spans
        # documents/dates, so chronological ordering is meaningless here.
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        page = ordered[args.offset : args.offset + args.limit]
        return _envelope({
            "count_by": args.count_by,
            "offset": args.offset,
            "limit": args.limit,
            "distinct_value_count": len(counter),
            "total_match_count": sum(counter.values()),
            "truncated": truncated,
            "groups": [{"value": value, "count": count} for value, count in page],
        })

    if count_mode == "document":
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
                _project_count_by_document(
                    {"chunk_id": rep_chunk_id, "document_id": source_id,
                     "file_name": file_name, "chunk_count": chunk_count},
                    args.returning,
                )
                for source_id, file_name, chunk_count, rep_chunk_id in cur.execute(
                    f"SELECT c.source_id, d.file_name, COUNT(*) AS n, "
                    f"       MIN(c.chunk_id) AS rep "
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

    # authored_date rides the summaries LEFT JOIN that --sort date already needs —
    # the canonical summarizer-inferred date that list_documents and the date
    # filters use — so it costs no extra query and is NULL for undated docs (no
    # summary, or a summary with no inferred date).
    rows = cur.execute(
        f"SELECT c.chunk_id, c.source_id, c.chunk_index, c.section_heading, "
        f"       c.page_number, c.content_type, c.text, d.file_name, "
        f"       s.authored_date "
        f"FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"JOIN documents d ON d.document_id = c.source_id "
        f"LEFT JOIN summaries s ON s.document_id = c.source_id "
        f"WHERE {where} "
        f"ORDER BY {_CHUNK_ORDER_BY[args.sort]} "
        f"LIMIT ? OFFSET ?",
        [*params, args.limit, args.offset],
    )

    matches = []
    for (chunk_id, source_id, chunk_index, section_heading,
         page_number, content_type, text, file_name, authored_date) in rows:
        # The full whitelisted row, built once. --returning projects from it;
        # otherwise --brief / default shape it (and only then is text truncated).
        full = {
            "chunk_id": chunk_id,
            "document_id": source_id,
            "file_name": file_name,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "section_heading": section_heading,
            "content_type": content_type,
            "authored_date": authored_date,
            "text": apply_preview(text, preview),
            "text_length": len(text),
        }
        projected = project_row(full, args.returning, MATCH_FIELDS)
        if projected is not None:
            matches.append(projected)
        elif args.brief:
            matches.append({
                "document_id": source_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "page_number": page_number,
                "authored_date": authored_date,
            })
        else:
            matches.append(full)
    return _envelope({
        "offset": args.offset, "limit": args.limit, "total": total,
        "preview": preview, "matches": matches,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="scan", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
