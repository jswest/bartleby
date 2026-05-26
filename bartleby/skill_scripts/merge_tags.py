#!/usr/bin/env python3
"""merge_tags — move all assignments from one tag onto another, then delete the source.

Errors if either tag doesn't exist or if ``--from`` == ``--to``.

Output:
    {"status": "merged", "from": {"tag_id": int, "name": str},
     "to": {"tag_id": int, "name": str}, "moved_assignments": int}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, run
from bartleby.skill_scripts._tags import get_tag_by_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="merge_tags")
    p.add_argument("--from", type=str, required=True, dest="from_name")
    p.add_argument("--to", type=str, required=True, dest="to_name")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if args.from_name == args.to_name:
        raise SkillError(
            "SELF_MERGE", "--from and --to must be different tags."
        )

    src = get_tag_by_name(conn, args.from_name)
    if src is None:
        raise SkillError("TAG_NOT_FOUND", f"No tag named {args.from_name!r}.")
    dst = get_tag_by_name(conn, args.to_name)
    if dst is None:
        raise SkillError("TAG_NOT_FOUND", f"No tag named {args.to_name!r}.")

    cur = conn.cursor()
    moved_before = cur.execute(
        "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (src.tag_id,),
    ).fetchone()[0]
    cur.execute(
        "INSERT OR IGNORE INTO document_tags (document_id, tag_id) "
        "SELECT document_id, ? FROM document_tags WHERE tag_id = ?",
        (dst.tag_id, src.tag_id),
    )
    cur.execute("DELETE FROM tags WHERE tag_id = ?", (src.tag_id,))
    return {
        "status": "merged",
        "from": {"tag_id": src.tag_id, "name": src.name},
        "to": {"tag_id": dst.tag_id, "name": dst.name},
        "moved_assignments": moved_before,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="merge_tags", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
