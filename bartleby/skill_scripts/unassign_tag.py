#!/usr/bin/env python3
"""unassign_tag — detach one tag from one document.

The inverse of ``assign_tag``. Removes a single ``(document, tag)`` assignment
without touching the tag itself — unlike ``delete_tag``, which drops the tag
and cascades *every* document's assignment. Use it to fix one mistaken or
stale assignment.

Single document, single tag. No-op when the assignment isn't present (the
``DELETE`` simply matches nothing); the call still succeeds.

Output:
    {"document_id": int, "file_name": str, "tag_id": int, "tag": str,
     "unassigned": true}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._tags import get_document, get_tag_by_name, unassign


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("unassign_tag", __doc__)
    p.add_argument("--document", type=int, required=True, dest="document_id")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    tag = get_tag_by_name(conn, args.tag)
    if tag is None:
        raise SkillError("TAG_NOT_FOUND", f"No tag named {args.tag!r}.")

    doc = get_document(conn, args.document_id)
    if doc is None:
        raise SkillError(
            "DOCUMENT_NOT_FOUND", f"No document with id {args.document_id}.",
        )
    document_id, file_name = doc

    unassign(conn, document_id, tag.tag_id)
    return {
        "document_id": document_id,
        "file_name": file_name,
        "tag_id": tag.tag_id,
        "tag": tag.name,
        "unassigned": True,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="unassign_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
