#!/usr/bin/env python3
"""read_tags — list the controlled tag vocabulary.

Output:
    {
      "tags": [
        {"tag_id": int, "name": str, "description": str, "doc_count": int}
      ]
    }

Always call this before any other tag operation so you know what already
exists. ``doc_count`` is the number of documents assigned each tag.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="read_tags")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    rows = conn.cursor().execute(
        "SELECT t.tag_id, t.name, t.description, "
        "       COALESCE(dt.n, 0) AS doc_count "
        "FROM tags t "
        "LEFT JOIN (SELECT tag_id, COUNT(*) AS n FROM document_tags "
        "           GROUP BY tag_id) dt "
        "  ON dt.tag_id = t.tag_id "
        "ORDER BY t.name"
    )
    return {
        "tags": [
            {"tag_id": tid, "name": n, "description": d, "doc_count": dc}
            for tid, n, d, dc in rows
        ],
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_tags", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
