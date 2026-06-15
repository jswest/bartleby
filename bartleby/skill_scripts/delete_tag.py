#!/usr/bin/env python3
"""delete_tag — remove a tag from the vocabulary.

The FK ``ON DELETE CASCADE`` on ``document_tags.tag_id`` clears all
assignments automatically.

Output:
    {"status": "deleted", "tag_id": int, "name": str,
     "removed_assignments": int}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run
from bartleby.skill_scripts._ids import format_output_ids
from bartleby.skill_scripts._tags import require_tag_by_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("delete_tag", __doc__)
    p.add_argument("--tag", type=str, required=True)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    tag = require_tag_by_name(conn, args.tag)

    cur = conn.cursor()
    n_assignments = cur.execute(
        "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (tag.tag_id,),
    ).fetchone()[0]
    cur.execute("DELETE FROM tags WHERE tag_id = ?", (tag.tag_id,))
    return format_output_ids({
        "status": "deleted",
        "tag_id": tag.tag_id,
        "name": tag.name,
        "removed_assignments": n_assignments,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="delete_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
