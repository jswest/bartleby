#!/usr/bin/env python3
"""annotate_finding — attach a note to a finding without rewriting it.

An annotation is commentary layered on a finding — a verified error, a caveat,
a pointer to better evidence. The finding body is never touched. Optionally
anchor the note to a verbatim span of the finding's raw markdown body with
``--quote`` (``--quote-prefix`` / ``--quote-suffix`` disambiguate a span that
repeats; both require ``--quote``), and optionally point it at a chunk with
``--chunk-id``. No ``--quote`` means a whole-finding note.

The note text comes from exactly one of ``--body`` or ``--body-file`` (read
verbatim, bypassing shell expansion — prefer it when the text may contain
``$``, backticks, or parens).

Output (every id is type-tagged):
    {
      "annotation_id": "annotation:<id>",
      "finding_id": "finding:<id>",
      "body": str,
      "anchor": {"exact": str, "prefix": str|null, "suffix": str|null}|null,
      "chunk_id": "chunk:<id>"|null,
      "is_human_author": false
    }

Errors: ``FINDING_NOT_FOUND``; ``ANCHOR_NOT_FOUND`` when ``--quote`` (with any
prefix/suffix) doesn't occur verbatim in the finding's current body;
``QUOTE_REQUIRED`` for an empty ``--quote`` or a prefix/suffix without one; ``UNKNOWN_CHUNK``;
``EMPTY_BODY`` / ``BODY_FILE_NOT_FOUND``. In a memory-off session you can only
annotate findings *this* session authored; annotating another session's
finding raises ``{"code": "MEMORY_OFF"}`` — an annotation quotes and attaches
to the finding's text, so it is gated exactly like the finding itself.
"""

from __future__ import annotations

import argparse

from bartleby.db.annotations import AnchorNotFound, insert_annotation
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import assert_findings_accessible, read_text_arg
from bartleby.skill_scripts._ids import format_output_ids, prefixed_int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("annotate_finding", __doc__)
    p.add_argument(
        "--finding-id", type=prefixed_int("finding"), required=True,
        dest="finding_id", help="Type-tagged finding id, e.g. finding:204.",
    )
    body_g = p.add_mutually_exclusive_group(required=True)
    body_g.add_argument("--body", type=str)
    body_g.add_argument("--body-file", dest="body_file")
    p.add_argument(
        "--quote", type=str, default=None,
        help="Verbatim span of the finding body to anchor the note to.",
    )
    p.add_argument("--quote-prefix", type=str, default=None, dest="quote_prefix")
    p.add_argument("--quote-suffix", type=str, default=None, dest="quote_suffix")
    p.add_argument(
        "--chunk-id", type=prefixed_int("chunk"), default=None, dest="chunk_id",
        help="Type-tagged chunk id (e.g. chunk:4192) the note points at.",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    body = read_text_arg(
        args.body, args.body_file, flag="body", error_code="BODY_FILE_NOT_FOUND",
    )
    if not body.strip():
        raise SkillError("EMPTY_BODY", "Annotation body is empty.")
    if args.quote is None and (
        args.quote_prefix is not None or args.quote_suffix is not None
    ):
        raise SkillError(
            "QUOTE_REQUIRED", "--quote-prefix/--quote-suffix require --quote.",
        )
    if args.quote == "":
        raise SkillError("QUOTE_REQUIRED", "--quote must be non-empty.")

    cur = conn.cursor()
    if cur.execute(
        "SELECT 1 FROM findings WHERE finding_id = ?", (args.finding_id,),
    ).fetchone() is None:
        raise SkillError(
            "FINDING_NOT_FOUND", f"No finding with id {args.finding_id}.",
        )

    # An annotation quotes and attaches to the finding's text, so it discloses
    # and extends that finding: visibility = the parent finding's (#689).
    assert_findings_accessible(conn, session_id, [args.finding_id], action="annotate")

    if args.chunk_id is not None and cur.execute(
        "SELECT 1 FROM chunks WHERE chunk_id = ?", (args.chunk_id,),
    ).fetchone() is None:
        raise SkillError(
            "UNKNOWN_CHUNK", f"No chunk with id chunk:{args.chunk_id}.",
        )

    try:
        annotation_id = insert_annotation(
            conn,
            finding_id=args.finding_id,
            body=body,
            is_human_author=False,
            anchor_exact=args.quote,
            anchor_prefix=args.quote_prefix,
            anchor_suffix=args.quote_suffix,
            chunk_id=args.chunk_id,
            session_id=session_id,
        )
    except AnchorNotFound:
        raise SkillError(
            "ANCHOR_NOT_FOUND",
            "--quote (with any --quote-prefix/--quote-suffix) does not occur "
            "verbatim in the finding's current body. Copy the span exactly from "
            "read_finding's body.",
        ) from None

    anchor = None
    if args.quote is not None:
        anchor = {
            "exact": args.quote,
            "prefix": args.quote_prefix,
            "suffix": args.quote_suffix,
        }
    return format_output_ids({
        "annotation_id": annotation_id,
        "finding_id": args.finding_id,
        "body": body,
        "anchor": anchor,
        "chunk_id": args.chunk_id,
        "is_human_author": False,
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="annotate_finding", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
