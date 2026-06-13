#!/usr/bin/env python3
"""read_document — read full text and/or summary for one document.

Returns both by default. `--summary` or `--full` narrow the response. The
full text is gated by ``max_read_tokens`` in the config; pass ``--force`` to
override.

Successful output:
    {
      "document": {
        "id": int, "file_name": str, "token_count": int,
        "authored_date": str|null
      },
      "summary": str|null,
      "full_text": str|null
    }

``authored_date`` is the document's date (lives on its summary row); it is
populated regardless of whether the summary text is real, because a
``backfill``-stub row carries a real date but no summary text. ``summary`` is
``null`` for such a stub (and for a document with no summary row at all).

DOCUMENT_TOO_LARGE error envelope (when --full would exceed max_read_tokens):
    {
      "error": "...",
      "code": "DOCUMENT_TOO_LARGE",
      "token_count": int,
      "max_read_tokens": int
    }
"""

from __future__ import annotations

import argparse

from bartleby.config import load_config
from bartleby.lib.consts import BACKFILL_MODEL
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import positive_int


DEFAULT_MAX_READ_TOKENS = 50_000


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_document", __doc__)
    p.add_argument("--document", type=positive_int, required=True, dest="document_id")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _reassemble_full_text(conn, document_id: int) -> str:
    rows = conn.cursor().execute(
        "SELECT text FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ? "
        "ORDER BY chunk_index",
        (document_id,),
    )
    return "\n\n".join(row[0] for row in rows)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    doc_row = cur.execute(
        "SELECT document_id, file_name, token_count "
        "FROM documents WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()
    if doc_row is None:
        raise SkillError(
            "DOCUMENT_NOT_FOUND",
            f"No document with id {args.document_id}.",
        )
    doc_id, file_name, token_count = doc_row

    want_summary = args.summary or not args.full
    want_full = args.full or not args.summary

    if want_full and not args.force:
        max_read_tokens = int(
            load_config().get("max_read_tokens", DEFAULT_MAX_READ_TOKENS)
        )
        if token_count is not None and token_count > max_read_tokens:
            raise SkillError(
                "DOCUMENT_TOO_LARGE",
                f"Document exceeds max_read_tokens ({max_read_tokens}). "
                "Pass --force to read anyway, or use read_chunks for paginated access.",
                token_count=token_count,
                max_read_tokens=max_read_tokens,
            )

    # Fetch the summary row regardless of mode: the date rides on it and is
    # exposed even when the text is suppressed or --full was requested. A
    # backfill stub carries a real date but no summary text, so its `text` ('')
    # is reported as null — the date is honest, the "summary" is not.
    row = cur.execute(
        "SELECT text, model, authored_date FROM summaries WHERE document_id = ?",
        (doc_id,),
    ).fetchone()
    authored_date = row[2] if row else None

    summary_text: str | None = None
    if want_summary and row is not None and row[1] != BACKFILL_MODEL:
        summary_text = row[0]

    full_text: str | None = None
    if want_full:
        full_text = _reassemble_full_text(conn, doc_id)

    return {
        "document": {
            "id": doc_id,
            "file_name": file_name,
            "token_count": token_count,
            "authored_date": authored_date,
        },
        "summary": summary_text,
        "full_text": full_text,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_document", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
