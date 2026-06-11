#!/usr/bin/env python3
"""read_tags — list the controlled tag vocabulary.

Output:
    {
      "tags": [
        {"tag_id": int, "name": str, "description": str, "doc_count": int,
         "value_type": str|null, "pattern": str|null}
      ]
    }

``value_type`` / ``pattern`` are non-null only for **value-tags** (a tag
carrying a per-document value extracted by a regex, created via ``add_tag
--value-type/--pattern`` and populated by ``extract``); they are null for
ordinary boolean category tags. Pass ``--boolean-only`` to scope the listing to
boolean tags (``value_type IS NULL``) — the natural set for browse/filter
contexts, since a value-tag is a field, not a category.

Always call this before any other tag operation so you know what already
exists. ``doc_count`` is the number of documents assigned each tag.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_tags", __doc__)
    p.add_argument(
        "--boolean-only", action="store_true", dest="boolean_only",
        help="List only boolean tags (value_type IS NULL), excluding value-tags.",
    )
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    where = "WHERE t.value_type IS NULL " if args.boolean_only else ""
    rows = conn.cursor().execute(
        "SELECT t.tag_id, t.name, t.description, t.value_type, t.pattern, "
        "       COALESCE(dt.n, 0) AS doc_count "
        "FROM tags t "
        "LEFT JOIN (SELECT tag_id, COUNT(*) AS n FROM document_tags "
        "           GROUP BY tag_id) dt "
        "  ON dt.tag_id = t.tag_id "
        f"{where}"
        "ORDER BY t.name"
    )
    return {
        "tags": [
            {
                "tag_id": tid, "name": n, "description": d,
                "value_type": vt, "pattern": p, "doc_count": dc,
            }
            for tid, n, d, vt, p, dc in rows
        ],
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_tags", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
