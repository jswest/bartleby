#!/usr/bin/env python3
"""assign_tag — attach one tag to one document, bypassing the classifier.

The companion to ``tag``, which decides membership with an LLM that reads the
document *summary*. Reach for ``assign_tag`` when membership was determined
out-of-band — a body scan, a deterministic rule, a human call — especially for
body-level properties (OCR quality, language, "contains tables") that leave no
trace in the summary the classifier sees.

Single document, single tag. Idempotent: re-assigning an existing pair is a
no-op (the underlying insert is ``OR IGNORE``).

Output:
    {"document_id": int, "file_name": str, "tag_id": int, "tag": str,
     "assigned": true}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, run
from bartleby.skill_scripts._tags import assign, get_document, get_tag_by_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="assign_tag")
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

    assign(conn, document_id, [tag.tag_id])
    return {
        "document_id": document_id,
        "file_name": file_name,
        "tag_id": tag.tag_id,
        "tag": tag.name,
        "assigned": True,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="assign_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
