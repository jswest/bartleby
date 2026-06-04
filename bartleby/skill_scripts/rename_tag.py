#!/usr/bin/env python3
"""rename_tag — change a tag's name (assignments preserved).

Errors if the new name already exists; use ``merge_tags`` to combine.

Output:
    {"status": "renamed", "tag_id": int, "old_name": str, "new_name": str}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._tags import get_tag_by_name, normalize_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("rename_tag", __doc__)
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    new_name = args.new.strip()
    if not new_name:
        raise SkillError("EMPTY_NAME", "New tag name must be non-empty.")
    if not normalize_name(new_name):
        raise SkillError(
            "EMPTY_NORMALIZED_NAME",
            "New tag name must contain at least one alphanumeric character.",
        )

    tag = get_tag_by_name(conn, args.old)
    if tag is None:
        raise SkillError("TAG_NOT_FOUND", f"No tag named {args.old!r}.")

    existing = get_tag_by_name(conn, new_name)
    if existing is not None and existing.tag_id != tag.tag_id:
        raise SkillError(
            "TAG_EXISTS",
            f"A tag named {new_name!r} already exists. "
            "Use merge_tags if you want to combine them.",
        )

    conn.cursor().execute(
        "UPDATE tags SET name = ? WHERE tag_id = ?", (new_name, tag.tag_id),
    )
    return {
        "status": "renamed",
        "tag_id": tag.tag_id,
        "old_name": tag.name,
        "new_name": new_name,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="rename_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
