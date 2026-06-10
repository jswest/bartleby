#!/usr/bin/env python3
"""merge_tags — move all assignments from one tag onto another, then delete the source.

Errors if either tag doesn't exist or if ``--from`` == ``--into``.

Output:
    {"status": "merged", "from": {"tag_id": int, "name": str},
     "into": {"tag_id": int, "name": str}, "inserted": int,
     "already_present": int}

``inserted`` is the number of source assignments actually copied onto the
destination tag. ``already_present`` is the number skipped because the
destination tag already carried that document (the overlap absorbed by
``INSERT OR IGNORE``). Their sum is the source tag's pre-merge assignment count.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._tags import require_tag_by_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("merge_tags", __doc__)
    p.add_argument("--from", type=str, required=True, dest="from_name")
    p.add_argument("--into", type=str, required=True, dest="into_name")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if args.from_name == args.into_name:
        raise SkillError(
            "SELF_MERGE", "--from and --into must be different tags."
        )

    src = require_tag_by_name(conn, args.from_name)
    dst = require_tag_by_name(conn, args.into_name)

    cur = conn.cursor()
    moved_before = cur.execute(
        "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (src.tag_id,),
    ).fetchone()[0]
    cur.execute(
        "INSERT OR IGNORE INTO document_tags (document_id, tag_id) "
        "SELECT document_id, ? FROM document_tags WHERE tag_id = ?",
        (dst.tag_id, src.tag_id),
    )
    inserted = conn.changes()
    cur.execute("DELETE FROM tags WHERE tag_id = ?", (src.tag_id,))
    return {
        "status": "merged",
        "from": {"tag_id": src.tag_id, "name": src.name},
        "into": {"tag_id": dst.tag_id, "name": dst.name},
        "inserted": inserted,
        "already_present": moved_before - inserted,
    }


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="merge_tags", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
