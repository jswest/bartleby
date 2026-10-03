#!/usr/bin/env python3
"""delete_annotation — remove one note from a finding.

The inverse of ``annotate_finding``. There is no in-place edit of an
annotation: to revise a note, delete it and annotate again. The finding itself
is untouched.

Output (every id is type-tagged):
    {
      "status": "deleted",
      "annotation_id": "annotation:<id>",
      "finding_id": "finding:<id>",
      "body": str
    }

``ANNOTATION_NOT_FOUND`` when the id doesn't exist. Annotations share their
parent finding's visibility: in a memory-off session you can only delete notes
on findings *this* session authored; a note on another session's finding raises
``{"code": "MEMORY_OFF"}`` (the response echoes the note, and the deletion
mutates that finding's record).
"""

from __future__ import annotations

import argparse

from bartleby.db.annotations import delete_annotation, get_annotation
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import assert_findings_accessible
from bartleby.skill_scripts._ids import format_output_ids, prefixed_int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("delete_annotation", __doc__)
    p.add_argument(
        "--annotation-id", type=prefixed_int("annotation"), required=True,
        dest="annotation_id", help="Type-tagged annotation id, e.g. annotation:12.",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    note = get_annotation(conn, args.annotation_id)
    if note is None:
        raise SkillError(
            "ANNOTATION_NOT_FOUND",
            f"No annotation with id annotation:{args.annotation_id}.",
        )

    # Visibility = the parent finding's (#689). Gate before the write/echo.
    assert_findings_accessible(
        conn, session_id, [note["finding_id"]], action="delete annotations on",
    )

    delete_annotation(conn, args.annotation_id)
    return format_output_ids({
        "status": "deleted",
        "annotation_id": args.annotation_id,
        "finding_id": note["finding_id"],
        "body": note["body"],
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="delete_annotation", parse_args=parse_args, work=work,
        argv=argv, mutates=True,
    )


if __name__ == "__main__":
    main()
