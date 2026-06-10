#!/usr/bin/env python3
"""add_tag — create a new tag in the controlled vocabulary.

Runs an embedding-similarity check against existing tag descriptions plus a
normalized-name exact-match check; on conflict, returns a non-error
``status: "conflict"`` envelope so the agent can surface it to the human
rather than silently creating a duplicate.

A plain boolean tag answers "which documents are X?"; pass --value-type and
--pattern together to create a **value-tag**, which carries a per-document
value extracted from chunk text (run `extract` afterward). --value-type is
one of number/string/date; --pattern is a regex with a named capture group
``(?P<value>…)`` marking the substring to extract (compiled with re2). Both
or neither — supplying one without the other is an error.

Output on success:
    {"status": "created", "tag": {"tag_id": int, "name": str,
                                  "description": str,
                                  "value_type": str|null, "pattern": str|null}}

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
from bartleby.skill_scripts._tags import (
    VALUE_TYPES,
    compile_pattern,
    find_similar_tag,
    normalize_name,
    validate_value_type,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("add_tag", __doc__)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--description", type=str, required=True)
    p.add_argument(
        "--value-type", dest="value_type", choices=list(VALUE_TYPES), default=None,
        help="Make this a value-tag casting captured text as this type. "
             "Requires --pattern.",
    )
    p.add_argument(
        "--pattern", type=str, default=None,
        help="Extraction regex with a (?P<value>…) group. Requires --value-type.",
    )
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

    # Value-tag: --value-type and --pattern are all-or-nothing. Validate the
    # type and compile the pattern (re2, must expose (?P<value>…)) before any
    # write so a malformed value-tag never lands.
    value_type = validate_value_type(args.value_type)
    pattern = args.pattern
    if (value_type is None) != (pattern is None):
        raise SkillError(
            "INCOMPLETE_VALUE_TAG",
            "--value-type and --pattern must be supplied together (or neither, "
            "for a plain boolean tag).",
        )
    if value_type is not None:
        compile_pattern(pattern)  # raises INVALID_PATTERN on bad regex/group

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
        "INSERT INTO tags (name, description, value_type, pattern) "
        "VALUES (?, ?, ?, ?)",
        (name, description, value_type, pattern),
    )
    tag_id = conn.last_insert_rowid()
    return {
        "status": "created",
        "tag": {
            "tag_id": tag_id, "name": name, "description": description,
            "value_type": value_type, "pattern": pattern,
        },
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="add_tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
