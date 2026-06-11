#!/usr/bin/env python3
"""save_date — backfill or correct a document's authored_date.

Output:
    {"document_id": int, "old_authored_date": str|null, "new_authored_date": str|null}

A curation write in the ``save_summary`` / ``assign_tag`` mold: use it when you
have *read the evidence* for the date (a dateline, a signed-on line, an export
header) and the metadata is wrong or missing. Cite-don't-guess.

``authored_date`` lives on the document's *summary* row — that is the column
every read path (``list_documents``, ``scan``/``search``, the
``--authored-after``/``--authored-before`` filters) actually reads. So this
script writes ``summaries.authored_date`` and the document immediately
participates in the date filters. A document with no summary has nowhere to
carry a date; run ``save_summary`` first.
"""

from __future__ import annotations

import argparse

from bartleby.ingest.summarize import normalize_authored_date
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import positive_int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("save_date", __doc__)
    p.add_argument("--document", type=positive_int, required=True, dest="document_id")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--date", type=str, default=None, dest="authored_date",
        help="ISO 8601 YYYY-MM-DD. Rejected (non-zero exit) if malformed.",
    )
    group.add_argument(
        "--clear", action="store_true",
        help="Set authored_date to NULL (mark the document undated).",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if args.clear:
        new_date = None
    else:
        # Unlike summarizer storage (which silently NULLs junk), a manual
        # curation write must reject malformed input loudly — a silent NULL
        # would look like a successful set but quietly clear the field.
        new_date = normalize_authored_date(args.authored_date)
        if new_date is None:
            raise SkillError(
                "INVALID_DATE",
                "--date must be a real calendar date in YYYY-MM-DD form; "
                f"got {args.authored_date!r}.",
            )

    cur = conn.cursor()
    row = cur.execute(
        "SELECT summary_id, authored_date FROM summaries WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()

    if row is None:
        doc = cur.execute(
            "SELECT document_id FROM documents WHERE document_id = ?",
            (args.document_id,),
        ).fetchone()
        if doc is None:
            raise SkillError(
                "DOCUMENT_NOT_FOUND",
                f"No document with id {args.document_id}.",
            )
        raise SkillError(
            "NO_SUMMARY",
            f"Document {args.document_id} has no summary to carry a date; "
            "run save_summary first (authored_date lives on the summary row).",
        )

    summary_id, old_date = row
    cur.execute(
        "UPDATE summaries SET authored_date = ? WHERE summary_id = ?",
        (new_date, summary_id),
    )

    return {
        "document_id": args.document_id,
        "old_authored_date": old_date,
        "new_authored_date": new_date,
    }


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="save_date", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
