#!/usr/bin/env python3
"""rename_tag — change a tag's name (assignments preserved).

Errors if the new name already exists — by the same normalized-name check
``add_tag`` runs, so ``"NYSEG"`` collides with an existing ``"ny-seg"`` rather
than creating a duplicate; use ``merge_tags`` to combine. A case/punctuation-
only self-rename of the same tag (e.g. ``"ny-seg"`` → ``"NYSEG"``) is allowed.

Output:
    {"status": "renamed", "tag_id": int, "old_name": str, "new_name": str}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._ids import format_output_ids
from bartleby.skill_scripts._tags import (
    find_tag_by_normalized_name, normalize_name, require_tag_by_name,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("rename_tag", __doc__)
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    new_name = args.new.strip()
    if not normalize_name(new_name):
        raise SkillError(
            "EMPTY_NORMALIZED_NAME",
            "New tag name must contain at least one alphanumeric character.",
        )

    tag = require_tag_by_name(conn, args.old)

    # tag_id guard lets a case/punctuation-only self-rename through.
    existing = find_tag_by_normalized_name(conn, new_name)
    if existing is not None and existing.tag_id != tag.tag_id:
        raise SkillError(
            "TAG_EXISTS",
            f"A tag named {existing.name!r} already exists "
            f"(normalized-equal to {new_name!r}). "
            "Use merge_tags if you want to combine them.",
        )

    conn.cursor().execute(
        "UPDATE tags SET name = ? WHERE tag_id = ?", (new_name, tag.tag_id),
    )
    return format_output_ids({
        "status": "renamed",
        "tag_id": tag.tag_id,
        "old_name": tag.name,
        "new_name": new_name,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="rename_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
