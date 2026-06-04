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
the null count would imply more temporal coverage than the corpus has.

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
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import run


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="describe_corpus")
    p.add_argument("--project", type=str, default=None)
    p.add_argument(
        "--top-n", type=_positive_int, default=5, dest="top_n",
        help="How many largest-by-token documents to list (default 5).",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()

    document_count = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    token_count = cur.execute(
        "SELECT COALESCE(SUM(token_count), 0) FROM documents"
    ).fetchone()[0]

    # Corpus content mix: ingested chunks only (document + image), grouped by
    # content_type (NULL = plain document text). chunk_count is their total.
    content_mix = [
        {"content_type": content_type, "chunk_count": n}
        for content_type, n in cur.execute(
            "SELECT content_type, COUNT(*) AS n FROM chunks "
            "WHERE source_kind IN ('document', 'image') "
            "GROUP BY content_type ORDER BY n DESC, content_type"
        )
    ]
    chunk_count = sum(row["chunk_count"] for row in content_mix)

    # Dates live on summaries; COUNT(authored_date) excludes NULLs. Undated
    # rolls in both NULL-date summaries and documents with no summary at all.
    min_date, max_date, dated_document_count = cur.execute(
        "SELECT MIN(authored_date), MAX(authored_date), COUNT(authored_date) "
        "FROM summaries"
    ).fetchone()

    documents_by_year = [
        {"year": year, "document_count": n}
        for year, n in cur.execute(
            "SELECT substr(authored_date, 1, 4) AS year, COUNT(*) AS n "
            "FROM summaries WHERE authored_date IS NOT NULL "
            "GROUP BY year ORDER BY year"
        )
    ]

    tags = [
        {"name": name, "document_count": n}
        for name, n in cur.execute(
            "SELECT t.name, COUNT(dt.document_id) AS n FROM tags t "
            "LEFT JOIN document_tags dt ON dt.tag_id = t.tag_id "
            "GROUP BY t.tag_id ORDER BY n DESC, t.name"
        )
    ]

    summarized = cur.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]

    largest_documents = [
        {"id": doc_id, "file_name": file_name, "title": title,
         "token_count": tok}
        for doc_id, file_name, title, tok in cur.execute(
            "SELECT d.document_id, d.file_name, s.title, d.token_count "
            "FROM documents d LEFT JOIN summaries s USING (document_id) "
            "ORDER BY (d.token_count IS NULL), d.token_count DESC, d.document_id "
            "LIMIT ?",
            (args.top_n,),
        )
    ]

    return {
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
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="describe_corpus", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
