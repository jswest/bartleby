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

Output (ids are type-tagged, e.g. ``"tag:7"``, ``"document:1"``):
    {"tag_id": "tag:<id>", "tag": str,
     "unassigned": [{"document_id": "document:<id>", "file_name": str}, ...],
     "not_found": ["document:<id>", ...]}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run
from bartleby.skill_scripts._ids import (
    format_id, format_output_ids, prefixed_int_list,
)
from bartleby.skill_scripts._tags import require_tag_by_name, resolve_documents, unassign


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("unassign_tag", __doc__)
    p.add_argument(
        "--documents", type=prefixed_int_list("document"), required=True,
        dest="document_ids",
        help="Comma-separated type-tagged document ids, e.g. document:1,document:2.",
    )
    p.add_argument("--tag", type=str, required=True)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    tag = require_tag_by_name(conn, args.tag)

    found, not_found = resolve_documents(conn, args.document_ids)
    for document_id, _ in found:
        unassign(conn, document_id, tag.tag_id)

    return format_output_ids({
        "tag_id": tag.tag_id,
        "tag": tag.name,
        "unassigned": [{"document_id": d, "file_name": f} for d, f in found],
        # not_found is a document-id list (not in the field map): tag each.
        "not_found": [format_id("document", d) for d in not_found],
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="unassign_tag", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
