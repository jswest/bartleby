#!/usr/bin/env python3
"""describe_corpus — one cheap aggregate overview of the whole corpus.

The recommended *first* call on an unfamiliar corpus. Where ``list_documents``
dumps rows (and on a large corpus that first call is huge), this answers the
agent's actual first question — "what *is* this corpus?" — with a single
compact summary. All plain SQL aggregates: no LLM, no embedding, sub-millisecond.

After reading it once ("689 docs, 2023–2026, all carry ch/nyseg tags, 12
missing summaries, mass clustered in 2024"), issue *targeted* ``list_documents``
calls (scoped by ``--tag`` / date) or jump straight to ``search``.

"Corpus" counts (``chunk_count`` / ``content_mix``) cover ingested material only
— ``source_kind IN ('document','image')``. Summary and finding chunks are
derived agent artifacts and are deliberately excluded.

Honesty note: ``authored_date`` is summarizer-inferred and silently stored NULL
on anything that isn't a clean ``YYYY-MM-DD``, so it's frequently absent. The
date range is reported *alongside* ``undated_document_count`` — a range without
the null count would imply more temporal coverage than the slice has.

Scope: the same flags ``search`` / ``scan`` / ``list_documents`` accept —
``--tag`` (repeatable, OR), ``--in-documents``, and the inclusive ``YYYY-MM-DD``
``--authored-after`` / ``--authored-before`` bounds (``--include-nulls`` keeps
undated docs). With any of them active, *every* aggregate is computed over that
filtered subset instead of the whole corpus, and a ``filters`` object echoes the
scope (``{tags, in_documents, authored_after, authored_before, include_nulls,
excluded_null_dated}``) so the numbers are self-describing. The ``tags`` facet
then lists only tags present in the slice. ``filters`` is absent on an
unfiltered call (the output is then identical to the whole-corpus overview).

Output:
    {
      "document_count": int,
      "chunk_count": int,            # document + image chunks only
      "token_count": int,            # SUM(documents.token_count)
      "authored_date": {
        "min": str|null, "max": str|null,
        "dated_document_count": int, "undated_document_count": int
      },
      "documents_by_year": [{"year": str, "document_count": int}, ...],
      "tags": [{"name": str, "document_count": int}, ...],
      "summary_coverage": {"summarized": int, "unsummarized": int},
      "content_mix": [{"content_type": str|null, "chunk_count": int}, ...],
      "largest_documents": [
        {"id": int, "file_name": str, "title": str|null, "token_count": int}, ...
      ]
      # plus "filters": {...} when a scope filter is active
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run
from bartleby.skill_scripts._common import (
    add_date_filter_args, comma_int_list, positive_int,
)
from bartleby.skill_scripts._tags import resolve_scope


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("describe_corpus", __doc__)
    p.add_argument("--project", type=str, default=None)
    p.add_argument(
        "--top-n", type=positive_int, default=5, dest="top_n",
        help="How many largest-by-token documents to list (default 5).",
    )
    p.add_argument(
        "--tag", action="append", default=None, dest="tags",
        help="Restrict the overview to documents carrying this tag. Repeat for "
             "OR semantics.",
    )
    p.add_argument(
        "--in-documents", type=comma_int_list("document_id"), default=None,
        dest="in_documents",
        help="Comma-separated document_ids to restrict the overview to.",
    )
    add_date_filter_args(p)
    return p.parse_args(argv)


def _empty_overview() -> dict:
    """The all-zero overview for a scope that matched no documents."""
    return {
        "document_count": 0,
        "chunk_count": 0,
        "token_count": 0,
        "authored_date": {
            "min": None, "max": None,
            "dated_document_count": 0, "undated_document_count": 0,
        },
        "documents_by_year": [],
        "tags": [],
        "summary_coverage": {"summarized": 0, "unsummarized": 0},
        "content_mix": [],
        "largest_documents": [],
    }


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    scope = resolve_scope(
        conn,
        in_documents=args.in_documents,
        tags=args.tags,
        authored_after=args.authored_after,
        authored_before=args.authored_before,
        include_nulls=args.include_nulls,
    )
    # S: None = whole corpus, [] = a filter matched nothing, else the slice.
    S = scope.document_ids

    if S is not None and not S:
        return scope.echo_into(_empty_overview())

    doc_ph = ",".join("?" * len(S)) if S is not None else None

    def doc_where(col: str) -> tuple[str, list]:
        """A ' WHERE <col> IN (...)' restriction to the slice, or '' corpus-wide."""
        pred, params = scope.restrict_in(col)
        return (f" WHERE {pred}", params) if pred else ("", [])

    # document_count / token_count come from the documents table directly so
    # unresolvable --in-documents ids (which never reach the documents) don't
    # inflate the count.
    w, wp = doc_where("document_id")
    document_count = cur.execute(
        f"SELECT COUNT(*) FROM documents{w}", wp
    ).fetchone()[0]
    token_count = cur.execute(
        f"SELECT COALESCE(SUM(token_count), 0) FROM documents{w}", wp
    ).fetchone()[0]

    # Content mix: ingested chunks only (document + image), grouped by
    # content_type (NULL = plain document text). Image chunks key on image_id,
    # so the slice reaches them through document_images.
    if S is None:
        cm_where, cm_params = "WHERE source_kind IN ('document', 'image')", []
    else:
        cm_where = (
            f"WHERE (source_kind = 'document' AND source_id IN ({doc_ph})) "
            f"OR (source_kind = 'image' AND source_id IN "
            f"(SELECT image_id FROM document_images WHERE document_id IN ({doc_ph})))"
        )
        cm_params = [*S, *S]
    content_mix = [
        {"content_type": content_type, "chunk_count": n}
        for content_type, n in cur.execute(
            f"SELECT content_type, COUNT(*) AS n FROM chunks {cm_where} "
            f"GROUP BY content_type ORDER BY n DESC, content_type",
            cm_params,
        )
    ]
    chunk_count = sum(row["chunk_count"] for row in content_mix)

    # Dates live on summaries; COUNT(authored_date) excludes NULLs. Undated
    # rolls in both NULL-date summaries and documents with no summary at all.
    w, wp = doc_where("document_id")
    min_date, max_date, dated_document_count = cur.execute(
        f"SELECT MIN(authored_date), MAX(authored_date), COUNT(authored_date) "
        f"FROM summaries{w}", wp
    ).fetchone()

    dy_pred, dy_params = scope.restrict_in("document_id")
    dy_where = "WHERE authored_date IS NOT NULL" + (f" AND {dy_pred}" if dy_pred else "")
    documents_by_year = [
        {"year": year, "document_count": n}
        for year, n in cur.execute(
            f"SELECT substr(authored_date, 1, 4) AS year, COUNT(*) AS n "
            f"FROM summaries {dy_where} GROUP BY year ORDER BY year",
            dy_params,
        )
    ]

    # Corpus-wide lists every tag (LEFT JOIN, zeros included). A slice instead
    # lists only the tags present in it (INNER JOIN over the slice's docs).
    if S is None:
        tags = [
            {"name": name, "document_count": n}
            for name, n in cur.execute(
                "SELECT t.name, COUNT(dt.document_id) AS n FROM tags t "
                "LEFT JOIN document_tags dt ON dt.tag_id = t.tag_id "
                "GROUP BY t.tag_id ORDER BY n DESC, t.name"
            )
        ]
    else:
        tags = [
            {"name": name, "document_count": n}
            for name, n in cur.execute(
                f"SELECT t.name, COUNT(*) AS n FROM tags t "
                f"JOIN document_tags dt ON dt.tag_id = t.tag_id "
                f"WHERE dt.document_id IN ({doc_ph}) "
                f"GROUP BY t.tag_id ORDER BY n DESC, t.name",
                list(S),
            )
        ]

    w, wp = doc_where("document_id")
    summarized = cur.execute(
        f"SELECT COUNT(*) FROM summaries{w}", wp
    ).fetchone()[0]

    w, wp = doc_where("d.document_id")
    largest_rows = cur.execute(
        f"SELECT d.document_id, d.file_name, s.title, d.token_count "
        f"FROM documents d LEFT JOIN summaries s USING (document_id){w} "
        f"ORDER BY (d.token_count IS NULL), d.token_count DESC, d.document_id "
        f"LIMIT ?",
        [*wp, args.top_n],
    )
    largest_documents = [
        {"id": doc_id, "file_name": file_name, "title": title, "token_count": tok}
        for doc_id, file_name, title, tok in largest_rows
    ]

    return scope.echo_into({
        "document_count": document_count,
        "chunk_count": chunk_count,
        "token_count": token_count,
        "authored_date": {
            "min": min_date,
            "max": max_date,
            "dated_document_count": dated_document_count,
            "undated_document_count": document_count - dated_document_count,
        },
        "documents_by_year": documents_by_year,
        "tags": tags,
        "summary_coverage": {
            "summarized": summarized,
            "unsummarized": document_count - summarized,
        },
        "content_mix": content_mix,
        "largest_documents": largest_documents,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="describe_corpus", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
