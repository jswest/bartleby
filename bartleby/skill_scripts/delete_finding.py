#!/usr/bin/env python3
"""delete_finding — retract a finding from the corpus.

The missing curation primitive: ``save_finding`` / ``edit_finding`` only ever
add or revise, so stale duplicates (zero-citation drafts, superseded
iterations) accumulate with no way out. This removes one finding outright.

Deletion touches only the finding's own rows — its body chunks (``source_kind
= 'finding'``, cleared from ``chunks`` / ``chunks_fts`` / ``chunks_vec`` via
``delete_chunks_for``) and its ``finding_citations`` (which cascade when the
``findings`` row is deleted). The cited *document* chunks are untouched:
findings are derivative hints, never evidence, so a deletion has no
referential fallout for the corpus.

Output:
    {
      "status": "deleted",
      "finding_id": int,
      "title": str,
      "removed_chunks": int,        # finding body chunks removed
      "removed_citations": int      # finding_citations rows removed
    }

``FINDING_NOT_FOUND`` when the id doesn't exist.
"""

from __future__ import annotations

import argparse

from bartleby.db.chunks import delete_chunks_for
from bartleby.skill_runner import SkillError, build_arg_parser, run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("delete_finding", __doc__)
    p.add_argument("--finding", type=int, required=True, dest="finding_id")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT title FROM findings WHERE finding_id = ?", (args.finding_id,),
    ).fetchone()
    if row is None:
        raise SkillError(
            "FINDING_NOT_FOUND", f"No finding with id {args.finding_id}.",
        )
    title = row[0]

    n_citations = cur.execute(
        "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
        (args.finding_id,),
    ).fetchone()[0]

    with conn:
        removed_chunks = delete_chunks_for(conn, "finding", args.finding_id)
        cur.execute(
            "DELETE FROM findings WHERE finding_id = ?", (args.finding_id,),
        )

    return {
        "status": "deleted",
        "finding_id": args.finding_id,
        "title": title,
        "removed_chunks": removed_chunks,
        "removed_citations": n_citations,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="delete_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
