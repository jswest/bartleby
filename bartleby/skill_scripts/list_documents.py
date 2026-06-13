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
      "hint": str|null               # set when more pages remain
      # plus "filters": {...} when a scope filter (--tag / --file-like / date bound) is active
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
their count is reported as ``excluded_null_dated`` inside the ``filters`` echo
(so a hidden slice is never silent). Pass ``--include-nulls`` to keep undated
documents in the result despite an active date bound.

Filename filtering: ``--file-like <pattern>`` (SQL ``LIKE`` — ``%`` = any run,
``_`` = one char) keeps only documents whose ``file_name`` matches. Repeatable;
the patterns OR together and the group ANDs with ``--tag`` / date bounds.

Whenever any scope filter is active the response carries a ``filters`` object
echoing it — ``{tags, in_documents, file_like, authored_after, authored_before,
include_nulls, excluded_null_dated}`` — the same nested contract ``search`` /
``scan`` / ``describe_corpus`` emit; it is absent on an unfiltered listing.

Value-tags: when a ``--tag`` names a value-tag (one carrying a per-document
value), each returned document gains a ``tag_values`` object mapping that tag's
name to ``{"value": str|null, "chunk_id": int|null}`` for the document — its
value chip. A document with no extracted value for the tag maps to null entries.
``tag_values`` is omitted (not empty) on rows when no value-tag is among the
filters and on ``--brief`` rows.

``--returning <field>,...`` projects each document to exactly the named fields,
in the order given, overriding the brief/default/verbose tier (the envelope —
``total`` / pagination / ``filters`` — is untouched). Selectable fields:
``document_id``, ``id``, ``file_name``, ``title``, ``description``,
``authored_date``, ``created_at``, ``has_summary``, ``image_count``,
``page_count``, ``token_count``, ``chunk_count``, ``tag_values``.
``document_id`` is an alias of the default row's ``id`` (same value, so a
citable id is always selectable); ``--returning`` can reach the verbose-tier
columns without ``--verbose`` and ``tag_values`` is ``null`` when no value-tag
is filtered. An unknown field returns an ``UNKNOWN_RETURNING_FIELD`` error
naming the valid set.
"""

from __future__ import annotations

import argparse

from bartleby.lib.consts import BACKFILL_MODEL
from bartleby.skill_runner import build_arg_parser, run
from bartleby.skill_scripts._common import (
    add_date_filter_args, add_file_like_arg, add_returning_arg, comma_int_list,
    nonneg_int, pagination_hint, positive_int, project_row, validate_returning,
)


# --returning whitelist. document_id leads so the citable id is always
# selectable (it's an alias of the default row's "id" — both carry the same
# value; "id" stays selectable too so the default contract's own field name
# works). The set is the union of the brief/default/verbose tiers plus the
# value-tag chip. tag_values is null off a row when no value-tag is filtered
# (same honest-null posture as the default, which omits it entirely).
DOCUMENT_FIELDS = [
    "document_id", "id", "file_name", "title", "description", "authored_date",
    "created_at", "has_summary", "image_count", "page_count", "token_count",
    "chunk_count", "tag_values",
]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("list_documents", __doc__)
    p.add_argument("--limit", type=positive_int, default=200)
    p.add_argument("--offset", type=nonneg_int, default=0)
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
        "--in-documents",
        type=comma_int_list("document_id"),
        default=None,
        dest="in_documents",
        help="Comma-separated document_ids to restrict the listing to.",
    )
    p.add_argument(
        "--tag",
        action="append", default=None, dest="tags",
        help=(
            "Filter to documents carrying this tag. Repeat for OR semantics "
            "(e.g. --tag ch --tag nyseg). Unknown tag names raise."
        ),
    )
    add_file_like_arg(p)
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
    add_returning_arg(p, DOCUMENT_FIELDS)
    return p.parse_args(argv)


# Maps --sort to an ORDER BY body. Every option ends on document_id so the order
# is total and pagination is stable across pages. Static (no user input reaches
# the SQL), so safe to interpolate.
_ORDER_BY = {
    "id": "d.document_id",
    "title": "COALESCE(s.title, d.file_name) COLLATE NOCASE, d.document_id",
    "date": "(s.authored_date IS NULL), s.authored_date DESC, d.document_id",
}


def _value_tag_values(conn, tag_names, doc_ids):
    """Per-document value chips for any value-tags among ``tag_names``.

    Returns ``{tag_name: {document_id: {"value": str|null, "chunk_id":
    int|null}}}`` for the value-tags in the filter, or ``None`` when no
    value-tag is filtered (so the caller omits ``tag_values`` entirely). Only
    value-tags (``value_type IS NOT NULL``) surface a chip — boolean tags don't.
    """
    if not tag_names or not doc_ids:
        return None
    cur = conn.cursor()
    name_ph = ",".join("?" * len(tag_names))
    value_tags = list(cur.execute(
        f"SELECT tag_id, name FROM tags "
        f"WHERE name IN ({name_ph}) AND value_type IS NOT NULL",
        tag_names,
    ))
    if not value_tags:
        return None
    tag_ids = [tid for tid, _ in value_tags]
    tag_ph = ",".join("?" * len(tag_ids))
    doc_ph = ",".join("?" * len(doc_ids))
    out = {name: {} for _, name in value_tags}
    by_id = dict(value_tags)
    for tag_id, document_id, value, chunk_id in cur.execute(
        f"SELECT tag_id, document_id, value, chunk_id FROM document_tags "
        f"WHERE tag_id IN ({tag_ph}) AND document_id IN ({doc_ph})",
        tag_ids + list(doc_ids),
    ):
        out[by_id[tag_id]][document_id] = {"value": value, "chunk_id": chunk_id}
    return out


def work(*, conn, args, session_id) -> dict:
    from bartleby.skill_scripts._tags import resolve_scope

    # Reject a typo'd --returning field up front, so a zero-document filter still
    # returns UNKNOWN_RETURNING_FIELD rather than a silent empty result.
    validate_returning(args.returning, DOCUMENT_FIELDS)
    cur = conn.cursor()
    scope = resolve_scope(
        conn,
        in_documents=args.in_documents,
        tags=args.tags,
        file_like=args.file_like,
        authored_after=args.authored_after,
        authored_before=args.authored_before,
        include_nulls=args.include_nulls,
    )

    pred, where_params = scope.restrict_in("d.document_id")
    where_clause = f"WHERE {pred} " if pred else ""

    total = cur.execute(
        f"SELECT COUNT(*) FROM documents d {where_clause}",
        where_params,
    ).fetchone()[0]

    rows = cur.execute(
        "SELECT d.document_id, d.file_name, d.page_count, d.token_count, d.created_at, "
        # A backfill stub (#536) carries a date but no real summary: its title/
        # description are empty and has_summary must read false. Suppress its
        # title/description to NULL here so coverage and the listed fields stay
        # honest; authored_date still rides along from the same row. The three
        # '?' bind BACKFILL_MODEL (leading, before any WHERE params).
        "       (CASE WHEN s.model = ? THEN NULL ELSE s.title END) AS summary_title, "
        "       (CASE WHEN s.model = ? THEN NULL ELSE s.description END) AS summary_description, "
        "       s.authored_date AS summary_authored_date, "
        "       (s.summary_id IS NOT NULL AND s.model != ?) AS has_summary, "
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
        [BACKFILL_MODEL, BACKFILL_MODEL, BACKFILL_MODEL,
         *where_params, args.limit, args.offset],
    )

    documents = []
    rows = list(rows)
    doc_ids = [r[0] for r in rows]
    value_tag_values = _value_tag_values(conn, args.tags, doc_ids)
    for (
        doc_id, file_name, page_count, token_count, created_at,
        title, description, authored_date, has_summary, chunk_count, image_count,
    ) in rows:
        if args.returning is not None:
            # The full whitelisted row, built once. tag_values is the doc's
            # value chip when a value-tag is filtered, else null.
            tag_values = None
            if value_tag_values is not None:
                tag_values = {
                    name: value_tag_values[name].get(
                        doc_id, {"value": None, "chunk_id": None}
                    )
                    for name in value_tag_values
                }
            documents.append(project_row({
                "document_id": doc_id,
                "id": doc_id,
                "file_name": file_name,
                "title": title,
                "description": description,
                "authored_date": authored_date,
                "created_at": created_at,
                "has_summary": bool(has_summary),
                "image_count": image_count,
                "page_count": page_count,
                "token_count": token_count,
                "chunk_count": chunk_count,
                "tag_values": tag_values,
            }, args.returning, DOCUMENT_FIELDS))
            continue
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
        if value_tag_values is not None:
            doc["tag_values"] = {
                name: value_tag_values[name].get(
                    doc_id, {"value": None, "chunk_id": None}
                )
                for name in value_tag_values
            }
        documents.append(doc)

    hint = pagination_hint(args.offset, len(documents), total)

    return scope.echo_into({
        "documents": documents,
        "total": total,
        "offset": args.offset,
        "limit": args.limit,
        "verbose": args.verbose,
        "hint": hint,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="list_documents", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
