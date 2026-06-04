#!/usr/bin/env python3
"""add_tag — create a new tag in the controlled vocabulary.

Runs an embedding-similarity check against existing tag descriptions plus a
normalized-name exact-match check; on conflict, returns a non-error
``status: "conflict"`` envelope so the agent can surface it to the human
rather than silently creating a duplicate.

Output on success:
    {"status": "created", "tag": {"tag_id": int, "name": str, "description": str}}

Output on conflict (still exit 0 — agent decides what to do):
    {
      "status": "conflict",
      "similar_to": {"tag_id": int, "name": str, "description": str,
                     "similarity": float},
      "proposed": {"name": str, "description": str}
    }

Conversation is the override: if the human wants the duplicate created
anyway, they direct the agent toward `rename_tag`, `merge_tags`, or simply
to accept the existing tag. There is no `--force` flag.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._tags import find_similar_tag, normalize_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("add_tag", __doc__)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--description", type=str, required=True)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    name = args.name.strip()
    description = args.description.strip()
    if not name:
        raise SkillError("EMPTY_NAME", "Tag name must be non-empty.")
    if not description:
        raise SkillError(
            "EMPTY_DESCRIPTION", "Tag description must be non-empty."
        )
    if not normalize_name(name):
        raise SkillError(
            "EMPTY_NORMALIZED_NAME",
            "Tag name must contain at least one alphanumeric character.",
        )

    conflict = find_similar_tag(conn, name=name, description=description)
    if conflict is not None:
        return {
            "status": "conflict",
            "similar_to": {
                "tag_id": conflict.tag_id,
                "name": conflict.name,
                "description": conflict.description,
                "similarity": conflict.similarity,
            },
            "proposed": {"name": name, "description": description},
        }

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tags (name, description) VALUES (?, ?)",
        (name, description),
    )
    tag_id = conn.last_insert_rowid()
    return {
        "status": "created",
        "tag": {"tag_id": tag_id, "name": name, "description": description},
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="add_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
