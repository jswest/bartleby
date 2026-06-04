#!/usr/bin/env python3
"""assign_tag — attach one tag to one or more documents, bypassing the classifier.

The companion to ``tag``, which decides membership with an LLM that reads the
document *summary*. Reach for ``assign_tag`` when membership was determined
out-of-band — a body scan, a deterministic rule, a human call — especially for
body-level properties (OCR quality, language, "contains tables") that leave no
trace in the summary the classifier sees.

One tag, any number of documents in a single process start (``--documents
1,2,3``) — so a corpus-wide tagging pass pays interpreter startup once, not
once per document. Idempotent: re-assigning an existing pair is a no-op (the
underlying insert is ``OR IGNORE``). An id with no document is reported in
``not_found`` and skipped; the rest of the batch still applies.

Output:
    {"tag_id": int, "tag": str,
     "assigned": [{"document_id": int, "file_name": str}, ...],
     "not_found": [int, ...]}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import comma_int_list
from bartleby.skill_scripts._tags import assign, get_tag_by_name, resolve_documents


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("assign_tag", __doc__)
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
        assign(conn, document_id, [tag.tag_id])

    return {
        "tag_id": tag.tag_id,
        "tag": tag.name,
        "assigned": [{"document_id": d, "file_name": f} for d, f in found],
        "not_found": not_found,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="assign_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
