#!/usr/bin/env python3
"""unassign_tag — detach one tag from one or more documents.

The inverse of ``assign_tag``. Removes ``(document, tag)`` assignments without
touching the tag itself — unlike ``delete_tag``, which drops the tag and
cascades *every* document's assignment. Use it to fix mistaken or stale
assignments.

One tag, any number of documents in a single process start (``--documents
1,2,3``). No-op for a pair that isn't assigned (the ``DELETE`` simply matches
nothing); the call still succeeds. An id with no document is reported in
``not_found`` and skipped; the rest of the batch still applies.

Output:
    {"tag_id": int, "tag": str,
     "unassigned": [{"document_id": int, "file_name": str}, ...],
     "not_found": [int, ...]}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import comma_int_list
from bartleby.skill_scripts._tags import get_tag_by_name, resolve_documents, unassign


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("unassign_tag", __doc__)
    p.add_argument(
        "--documents", type=comma_int_list("document_id"), required=True,
        dest="document_ids", help="Comma-separated document ids, e.g. 1,2,3.",
    )
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    tag = get_tag_by_name(conn, args.tag)
    if tag is None:
        raise SkillError("TAG_NOT_FOUND", f"No tag named {args.tag!r}.")

    found, not_found = resolve_documents(conn, args.document_ids)
    for document_id, _ in found:
        unassign(conn, document_id, tag.tag_id)

    return {
        "tag_id": tag.tag_id,
        "tag": tag.name,
        "unassigned": [{"document_id": d, "file_name": f} for d, f in found],
        "not_found": not_found,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="unassign_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
