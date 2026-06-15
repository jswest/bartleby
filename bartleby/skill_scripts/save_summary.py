#!/usr/bin/env python3
"""save_summary — write (or replace) the agent-authored summary for a document.

Output:
    {"summary_id": "summary:<id>", "document_id": "document:<id>", "chunk_ids": ["chunk:<id>", ...]}

Title and description are required so summaries written by the agent show up
in ``list_documents`` the same way ingest-time summaries do.

Replacing an existing summary without passing ``--authored-date`` carries the
prior row's ``authored_date`` forward rather than nulling it, so a re-save
doesn't silently drop the document out of every authored-date scope. A brand-new
save with no prior row and no arg stays NULL; an explicit ``--authored-date``
always overwrites.
"""

from __future__ import annotations

import argparse

from bartleby.db.chunks import delete_chunks_for, insert_summary_chunks
from bartleby.ingest.summarize import normalize_authored_date
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import embed_body_chunks
from bartleby.skill_scripts._ids import format_id, format_output_ids, prefixed_int


_AUTHOR_MODEL = "agent"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("save_summary", __doc__)
    p.add_argument(
        "--document-id", type=prefixed_int("document"), required=True,
        dest="document_id", help="Type-tagged document id, e.g. document:204.",
    )
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--description", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument(
        "--authored-date", type=str, default=None, dest="authored_date",
        help=(
            "ISO 8601 YYYY-MM-DD. Silently stored as NULL if malformed. "
            "On replace, omitting this carries the prior summary's "
            "authored_date forward; on a brand-new save it stays NULL."
        ),
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if not args.title or not args.title.strip():
        raise SkillError("EMPTY_TITLE", "Summary title must be non-empty.")
    if not args.description or not args.description.strip():
        raise SkillError(
            "EMPTY_DESCRIPTION", "Summary description must be non-empty."
        )
    if not args.text or not args.text.strip():
        raise SkillError("EMPTY_TEXT", "Summary text must be non-empty.")

    cur = conn.cursor()
    doc_row = cur.execute(
        "SELECT document_id FROM documents WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()
    if doc_row is None:
        raise SkillError(
            "DOCUMENT_NOT_FOUND",
            f"No document with id {args.document_id}.",
        )

    prior = cur.execute(
        "SELECT summary_id, authored_date FROM summaries WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()

    # On replace, omit --authored-date to carry the prior row's date forward
    # rather than nulling it: a full-row REPLACE would otherwise silently drop
    # the doc out of every authored-date scope (issue #467). A brand-new save
    # with no prior row and no arg stays NULL; an explicit --authored-date
    # always overwrites.
    if args.authored_date is None and prior:
        authored_date = prior[1]
    else:
        authored_date = normalize_authored_date(args.authored_date)

    # Embed BEFORE the first write (the prior-summary delete below) so the
    # transaction's write lock doesn't span the lazy model load (issue #340 —
    # apsw's txn is deferred, so no lock is held until the first DELETE/INSERT).
    # On a failed replace this also means the prior summary and its chunks are
    # still untouched if embedding raises. Same chunk+embed as the finding path.
    chunk_inputs = embed_body_chunks(args.text)

    if prior:
        delete_chunks_for(conn, "summary", prior[0])
        cur.execute("DELETE FROM summaries WHERE summary_id = ?", (prior[0],))

    cur.execute(
        "INSERT INTO summaries "
        "(document_id, title, description, text, model, authored_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (args.document_id, args.title, args.description, args.text,
         _AUTHOR_MODEL, authored_date),
    )
    summary_id = conn.last_insert_rowid()

    chunk_ids = insert_summary_chunks(conn, summary_id, chunk_inputs)

    return format_output_ids({
        # summary_id is a summary id (not in the field map): tag it explicitly.
        "summary_id": format_id("summary", summary_id),
        "document_id": args.document_id,
        "chunk_ids": chunk_ids,
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="save_summary", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
