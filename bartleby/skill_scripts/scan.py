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
    The FTS query is confined to the chunk **body text** (the MATCH is
    column-qualified ``{text} : (...)``), so a term that appears only in a
    chunk's ``section_heading`` is *not* a match — every returned snippet
    actually contains the query. To filter on headings deliberately, use
    ``--heading-like`` (a separate ``section_heading LIKE`` predicate).

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
      "offset": int, "limit": int, "total": int,
      "preview": int,
      "filters": {   # present ONLY when a scope filter is active (see below)
        "tags": [str, ...], "in_documents": [int, ...],
        "file_like": [str, ...], "heading_like": [str, ...],
        "authored_after": str, "authored_before": str,
        "include_nulls": bool, "excluded_null_dated": int
      },
      "matches": [{
        "document_id": "document:<id>",
        "file_name": str,
        "chunk_id": "chunk:<id>",
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
        {"document_id": "document:<id>", "file_name": str, "chunk_count": int}, ...
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
    deadline (``CAPTURE_TIMEOUT`` on overrun) and a match cap; hitting the cap
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

Extract capture table (``--extract '/regex/'``, repeatable):
    The tidy-table sibling of ``--count-by '/regex/'``: instead of folding
    captures into a histogram, return **one row per matching chunk** carrying
    ``chunk_id`` / ``document_id`` / ``file_name`` plus every pattern's capture
    columns. Column naming is fixed by the capture: a **named** group
    ``(?P<amount>...)`` becomes a column ``amount``; **bare** groups become
    positional ``g1``, ``g2``, ... A pattern that doesn't match a given chunk
    yields ``null`` for its column(s) **without dropping the row** — so a row is
    a chunk, not a match. Repeat ``--extract`` to widen the table; column names
    across patterns must stay distinct (``EXTRACT_COLUMN_COLLISION`` otherwise —
    use ``(?P<name>...)`` to disambiguate). Only the **first** match per chunk is
    captured (this is a projection, not an enumeration; ``--count-by`` is the
    per-match fold). Composes with every scope (``--file-like`` / ``--heading-
    like`` / ``--in-documents`` / ``--tag`` / date bounds). Paginated like the
    default mode with an honest ``total``; rows are in ``(document_id,
    chunk_index)`` order. Cannot combine with ``--count-by`` / ``--preview`` /
    ``--brief`` / ``--returning`` (the capture columns *are* the projection).
    Deliberately **not** a query engine — no predicates, no joins, no
    aggregation; it returns a table for the agent to post-process. Shares the
    capture machinery and the runaway guards (``CAPTURE_TIMEOUT``) with
    ``--count-by``.

        scan "Income reported" --extract '/reported:\\s*\\$(?P<amount>[\\d,]+)/'
        → columns: ["amount"],
          rows: [{"chunk_id": "chunk:12", "document_id": "document:3",
                  "file_name": "f1.txt", "amount": "120,000"}, ...]

    {
      "query": str, "match_mode": "phrase" | "terms",
      "offset": int, "limit": int, "total": int,
      "columns": [str, ...],   # the capture column names, in pattern order
      "rows": [
        {"chunk_id": "chunk:<id>", "document_id": "document:<id>", "file_name": str,
         "<col>": str | null, ...}, ...
      ]
    }

Zero-result diagnosis:
    When a scan returns nothing — no matches, an empty histogram, an empty
    capture table — the envelope carries a ``diagnosis`` block (computed with a
    handful of cheap COUNTs fired **only** on that zero path, never on the hot
    path) that distinguishes "absent" from "present on a surface scan doesn't
    match". scan matches body ``text`` only, column-qualified, so a ``0`` is
    ambiguous: the term may live in *other documents* outside the scope, in a
    chunk's ``section_heading`` (which scan never matches), or only *semantically*
    (no literal token — only ``search`` can confirm that).

    {
      "diagnosis": {
        "verdict": "absent" | "out_of_scope" | "heading_only" | "filtered_out",
        "body_in_scope": int,        # body matches inside the active scope
        "body_corpus_wide": int,     # body matches ignoring every scope filter
        "heading_in_scope": int,     # section_heading matches inside the scope
        "heading_corpus_wide": int,  # section_heading matches corpus-wide
        "hint": str                  # points at --heading-like / search
      }
    }

    ``verdict`` reads the cheap signal: ``absent`` (no body/heading hit anywhere),
    ``out_of_scope`` (body hits exist but only outside this scope),
    ``heading_only`` (the term lives only in headings — reach it with
    ``--heading-like`` or ``search``), ``filtered_out`` (body hits exist *in
    scope* but a chunk-level filter like ``--heading-like`` dropped them). It is a
    hint, not a guarantee: a purely-semantic presence is invisible to these
    COUNTs and only confirmable with ``search``. An empty query carries no
    diagnosis (there is no FTS expression to count). The diagnosis keys off the
    full unpaginated ``total`` (not the current page), so a deep empty page past
    real matches — where ``total`` is still > 0 — carries no diagnosis: there
    are matches, just not on this page.

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
    CaptureSpec, add_date_filter_args, add_file_like_arg, add_returning_arg,
    apply_preview, nonneg_int, parse_capture_regex,
    positive_int, project_row, text_qualified_fts, validate_returning,
)
from bartleby.skill_scripts._ids import format_output_ids, prefixed_int_list
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

# Runaway guards for the regex-capture modes (--extract and --count-by '/regex/').
# The pattern is agent-supplied, not attacker-supplied, so this is a footgun rail,
# not an adversarial-ReDoS defence: the deadline is checked *between chunks* (a
# single re.finditer/search call is C code that can't be interrupted mid-match),
# which catches the realistic "broad pattern × big corpus" runaway. Chunks are
# size-bounded by the chunker, so the residual single-chunk-backtracking gap stays
# improbable. The match cap bounds Counter growth in --count-by; when hit, the
# result is flagged truncated rather than dropped silently.
CAPTURE_DEADLINE_SECONDS = 5.0
CAPTURE_MAX_MATCHES = 100_000


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
        type=prefixed_int_list("document"),
        default=None,
        dest="in_documents",
        help="Comma-separated type-tagged document ids to restrict the scan to "
             "(e.g. document:12,document:34).",
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
        "--extract",
        action="append", default=None, dest="extract", metavar="/regex/",
        help="Capture regex groups into columns. Per matching chunk, emit "
             "chunk_id/document_id/file_name plus each pattern's capture "
             "group(s): named (?P<x>...) groups become column 'x', bare groups "
             "become 'g1','g2',...; a non-matching pattern yields null columns "
             "without dropping the row. Repeat to widen the table (column names "
             "must stay distinct). Composes with all scopes; not a query engine "
             "— returns a tidy table to post-process. Excludes --count-by.",
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
    if args.extract is not None:
        if args.count_by is not None:
            p.error(
                "--extract and --count-by are different aggregations of the same "
                "capture machinery; pass one. (--count-by is extract-then-group-"
                "and-count; --extract returns the per-chunk capture table.)"
            )
        if args.preview is not None or args.brief:
            p.error(
                "--extract returns a per-chunk capture table whose columns are "
                "the regex groups, so it cannot be combined with --preview or "
                "--brief (which shape the default text snippet)."
            )
        if args.returning is not None:
            p.error(
                "--extract owns its output columns (the capture groups plus "
                "chunk_id/document_id/file_name), so --returning does not apply."
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


def _parse_count_by(value: str) -> tuple[str, CaptureSpec | None]:
    """Classify a --count-by value.

    Returns ('document', None) for the literal keyword, or ('regex', spec) for a
    /pattern/ that carries at least one capture group — parsed through the shared
    capture machinery (:func:`parse_capture_regex`), so --count-by '/regex/' is
    extract-then-group-and-count over the *same* primitive --extract uses. A
    capture spec may carry several groups; --count-by buckets on the first one.
    Raises SkillError on a malformed value, an uncompilable pattern, or a
    capture-less regex.
    """
    if value == "document":
        return "document", None
    if value.startswith("/"):
        return "regex", parse_capture_regex(value, flag="--count-by")
    raise SkillError(
        "INVALID_COUNT_BY",
        f"--count-by must be 'document' or a /regex/ with a capture group; "
        f"got {value!r}.",
    )


def _resolve_extract_columns(specs: list[CaptureSpec]) -> list[str]:
    """The ordered union of every --extract spec's columns, collisions rejected.

    Several --extract patterns project onto one chunk row, so their column names
    must be distinct. A repeated name (two bare groups both ``g1``, or two named
    groups sharing a name) would silently clobber a cell, so reject it with a
    clean envelope pointing at named groups as the fix rather than guessing which
    capture wins.
    """
    columns: list[str] = []
    for spec in specs:
        for col in spec.columns:
            if col in columns:
                raise SkillError(
                    "EXTRACT_COLUMN_COLLISION",
                    f"--extract column {col!r} is produced by more than one "
                    "pattern. Give each capture a distinct (?P<name>...) so its "
                    "column is unambiguous.",
                    column=col,
                )
            columns.append(col)
    return columns


def _count_total(cur, where: str, params: list) -> int:
    """Total matching document chunks (the unpaginated ``total``) for ``where``.

    Shared by the default and --extract per-chunk modes — both paginate the same
    chunk set and report its honest pre-pagination size.
    """
    return cur.execute(
        f"SELECT COUNT(*) FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"WHERE {where}",
        params,
    ).fetchone()[0]


def _scan_chunk_texts(cur, where: str, params: list, deadline_label: str):
    """Yield each matching chunk's ``(chunk_id, source_id, file_name, text)``,
    enforcing the shared between-chunk wall-clock deadline.

    The single chunk-text walk behind both regex modes (--extract and
    --count-by '/regex/'). The deadline is checked *between* chunks — a single
    re.finditer/search is C code that can't be interrupted mid-match — which
    catches the realistic "broad pattern × big corpus" runaway (see the
    CAPTURE_* guard note above). ``deadline_label`` names the flag in the
    timeout message.
    """
    deadline = time.monotonic() + CAPTURE_DEADLINE_SECONDS
    rows = cur.execute(
        f"SELECT c.chunk_id, c.source_id, d.file_name, c.text FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"JOIN documents d ON d.document_id = c.source_id "
        f"WHERE {where} "
        f"ORDER BY c.source_id, c.chunk_index",
        params,
    )
    for chunk_id, source_id, file_name, text in rows:
        if time.monotonic() > deadline:
            raise SkillError(
                "CAPTURE_TIMEOUT",
                f"{deadline_label} exceeded {CAPTURE_DEADLINE_SECONDS:g}s; "
                "narrow the scan scope or simplify the pattern.",
            )
        yield chunk_id, source_id, file_name, text


def _count_by_regex(cur, where: str, params: list, spec: CaptureSpec) -> tuple[Counter, bool]:
    """Tally spec's first capture group across every matching chunk's text.

    Extract-then-group-and-count: per match (a chunk with two matches contributes
    two), skipping a capture that didn't participate (group(1) is None). Returns
    the histogram and whether the match cap truncated it. The first column is the
    bucket key — --count-by is single-valued by contract, so extra groups in the
    spec are ignored here (they exist for --extract's wider table).
    """
    counter: Counter = Counter()
    total = 0
    for _chunk_id, _source_id, _file_name, text in _scan_chunk_texts(
        cur, where, params, "--count-by regex"
    ):
        for match in spec.pattern.finditer(text):
            value = match.group(1)  # group 1 == columns[0], the bucket key
            if value is None:
                continue
            counter[value] += 1
            total += 1
            if total >= CAPTURE_MAX_MATCHES:
                return counter, True  # match cap hit → partial counts
    return counter, False


def _heading_qualified_fts(fts_expr: str) -> str:
    """Column-qualify an FTS5 MATCH expression to the ``section_heading`` column.

    The heading-side mirror of :func:`text_qualified_fts`: ``{section_heading} :
    (<expr>)`` confines the diagnosis COUNT to chunk headings, so it counts only
    where the term lives in a heading and never in the body. Returns the
    expression unchanged when empty.
    """
    if not fts_expr:
        return fts_expr
    return f"{{section_heading}} : ({fts_expr})"


def _count_match(cur, column_fts: str, restrict: list | None) -> int:
    """COUNT document chunks whose ``column_fts`` MATCH holds, optionally scoped.

    A single cheap COUNT over ``chunks_fts`` joined to ``chunks`` — the diagnosis
    primitive. ``restrict`` is the scope's resolved document-id set (``None`` =
    whole corpus); ``column_fts`` is an already-column-qualified FTS expression
    (body via :func:`text_qualified_fts`, heading via :func:`_heading_qualified_fts`).
    Heading-like / pagination never reach here — the diagnosis answers "where does
    the signal live", not "what did this exact filtered scan return".
    """
    where = "chunks_fts MATCH ? AND c.source_kind = 'document'"
    params: list = [column_fts]
    if restrict is not None:
        placeholders = ",".join("?" * len(restrict))
        where += f" AND c.source_id IN ({placeholders})"
        params.extend(restrict)
    return cur.execute(
        f"SELECT COUNT(*) FROM chunks_fts "
        f"JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        f"WHERE {where}",
        params,
    ).fetchone()[0]


def _build_diagnosis(cur, fts_query: str, restrict: list | None) -> dict:
    """Coverage-aware diagnosis for a zero-result scan (fired only on that path).

    scan matches body ``text`` only, column-qualified — so a ``0`` can mean
    "absent" OR "present on a surface scan doesn't match". A handful of cheap
    COUNTs distinguish those cases without touching the hot path:

    - ``body_in_scope`` — body matches inside the active scope (with the
      ``--heading-like`` chunk filter and pagination stripped). > 0 here means the
      filtered scan returned 0 only because ``--heading-like`` (or another
      chunk-level narrowing) excluded the body hits, not because the term is absent.
    - ``body_corpus_wide`` — body matches ignoring every scope filter. > 0 while
      ``body_in_scope`` is 0 means the term is present in *other documents* outside
      the scope.
    - ``heading_in_scope`` / ``heading_corpus_wide`` — the same split for
      ``section_heading`` matches, which scan's body-qualified MATCH never returns.
      > 0 means the term lives in a heading; reach it with ``--heading-like`` or
      ``search`` (whose semantic leg contextualizes headings into the embedding).

    ``verdict`` summarizes the cheap signal: ``absent`` (nothing anywhere),
    ``out_of_scope`` (body hits exist but only outside the scope),
    ``heading_only`` (only headings carry the term), or ``filtered_out`` (body
    hits exist in scope but a chunk-level filter dropped them). It is a hint, not
    a guarantee — only ``search`` can confirm a purely-*semantic* presence, which
    scan cannot compute cheaply; that path is noted in ``hint``.
    """
    body_fts = text_qualified_fts(fts_query)
    heading_fts = _heading_qualified_fts(fts_query)

    body_corpus_wide = _count_match(cur, body_fts, None)
    heading_corpus_wide = _count_match(cur, heading_fts, None)
    # "in scope" follows the *document-level* restrict only; --heading-like is a
    # chunk-level filter and is deliberately stripped here, so body_in_scope > 0
    # on a zero scan means that filter (not absence) dropped the body hits.
    if restrict is None:
        # No document-level scope: in-scope equals corpus-wide, no extra COUNTs.
        body_in_scope, heading_in_scope = body_corpus_wide, heading_corpus_wide
    elif restrict:
        body_in_scope = _count_match(cur, body_fts, restrict)
        heading_in_scope = _count_match(cur, heading_fts, restrict)
    else:
        # Scope resolved to no documents — nothing in scope can match.
        body_in_scope = heading_in_scope = 0

    if body_in_scope:
        verdict = "filtered_out"  # body hits in scope, dropped by a chunk filter
    elif body_corpus_wide:
        verdict = "out_of_scope"  # body hits exist, just not in this scope
    elif heading_in_scope or heading_corpus_wide:
        verdict = "heading_only"  # term lives only in headings
    else:
        verdict = "absent"  # no body/heading hit anywhere in the corpus

    return {
        "verdict": verdict,
        "body_in_scope": body_in_scope,
        "body_corpus_wide": body_corpus_wide,
        "heading_in_scope": heading_in_scope,
        "heading_corpus_wide": heading_corpus_wide,
        "hint": (
            "scan matches body text only. Headings are reachable via "
            "--heading-like or search; a purely-semantic presence (no literal "
            "token) is only confirmable with search."
        ),
    }


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
    extract_specs = None
    if args.extract is not None:
        extract_specs = [
            parse_capture_regex(value, flag="--extract") for value in args.extract
        ]
        extract_columns = _resolve_extract_columns(extract_specs)
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
        # Type-tag every id field (document_id/chunk_id in matches, count-by docs,
        # extract rows, and the filters echo's in_documents). The diagnosis block
        # added later by _with_diagnosis carries only counts, no ids.
        return format_output_ids(env)

    fts_query = _build_fts_query(args.query, match_mode)
    # Nothing can match: an empty token set, or a scope that resolved to no
    # documents (an empty tag / in-documents / date slice).
    no_match = not fts_query or (restrict is not None and not restrict)

    # Confine MATCH to the body text (``{text} : (...)``) so a heading-only term
    # never inflates grep-totals with a snippet that doesn't contain it.
    # Deliberate heading recall stays available via --heading-like.
    where = "chunks_fts MATCH ? AND c.source_kind = 'document'"
    params: list = [text_qualified_fts(fts_query)]
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

    def _with_diagnosis(env: dict) -> dict:
        """Attach the coverage-aware ``diagnosis`` block — only fired here, on a
        confirmed zero result, so the extra COUNTs stay off the hot path. An empty
        token set carries no FTS expression to COUNT, so it gets no diagnosis."""
        if fts_query:
            env["diagnosis"] = _build_diagnosis(cur, fts_query, restrict)
        return env

    if extract_specs is not None:
        # Per matching chunk: chunk_id/document_id/file_name plus each spec's
        # captured columns (first match per spec; null cells for a non-matching
        # spec, never dropping the row). Paginated like default mode with an
        # honest `total`; the column set is echoed so the table is self-describing.
        if no_match:
            return _with_diagnosis(_envelope({
                "offset": args.offset, "limit": args.limit, "total": 0,
                "columns": extract_columns, "rows": [],
            }))
        total = _count_total(cur, where, params)
        rows_out = []
        page_end = args.offset + args.limit
        for i, (chunk_id, source_id, file_name, text) in enumerate(
            _scan_chunk_texts(cur, where, params, "--extract regex")
        ):
            # Page in Python over the deterministic scan order — the spec walk is
            # already a full read for the deadline; slicing here keeps one path.
            if i < args.offset:
                continue
            if i >= page_end:
                break
            row = {
                "chunk_id": chunk_id,
                "document_id": source_id,
                "file_name": file_name,
            }
            for spec in extract_specs:
                row.update(spec.extract_first(text))
            rows_out.append(row)
        env = _envelope({
            "offset": args.offset, "limit": args.limit, "total": total,
            "columns": extract_columns, "rows": rows_out,
        })
        return _with_diagnosis(env) if total == 0 else env

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
        env = _envelope({
            "count_by": args.count_by,
            "offset": args.offset,
            "limit": args.limit,
            "distinct_value_count": len(counter),
            "total_match_count": sum(counter.values()),
            "truncated": truncated,
            "groups": [{"value": value, "count": count} for value, count in page],
        })
        return _with_diagnosis(env) if not counter else env

    if count_mode == "document":
        validate_returning(args.returning, COUNT_BY_DOCUMENT_FIELDS)
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
        env = _envelope({
            "count_by": "document",
            "offset": args.offset,
            "limit": args.limit,
            "distinct_document_count": distinct_document_count,
            "total_chunk_count": total_chunk_count,
            "documents": documents,
        })
        return _with_diagnosis(env) if total_chunk_count == 0 else env

    # Default mode: paginated per-chunk matches.
    validate_returning(args.returning, MATCH_FIELDS)
    preview = args.preview if args.preview is not None else DEFAULT_PREVIEW
    total = 0 if no_match else _count_total(cur, where, params)
    if total == 0:
        return _with_diagnosis(_envelope({
            "offset": args.offset, "limit": args.limit, "total": 0,
            "preview": preview, "matches": [],
        }))

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
