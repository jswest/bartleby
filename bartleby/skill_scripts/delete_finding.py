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

``FINDING_NOT_FOUND`` when the id doesn't exist. In a memory-off session you
can only delete findings *this* session authored; deleting a finding written by
another session raises ``{"code": "MEMORY_OFF"}`` — the response echoes the
deleted title (a content reveal) and the deletion mutates another session's
work, both walled off to avoid contaminating an evaluation run.
"""

from __future__ import annotations

import argparse

from bartleby.db.chunks import delete_chunks_for
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import memory_enabled


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("delete_finding", __doc__)
    p.add_argument("--finding", type=int, required=True, dest="finding_id")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT session_id, title FROM findings WHERE finding_id = ?",
        (args.finding_id,),
    ).fetchone()
    if row is None:
        raise SkillError(
            "FINDING_NOT_FOUND", f"No finding with id {args.finding_id}.",
        )
    owning_session_id, title = row

    # The response echoes the deleted title, and the deletion mutates the
    # finding outright. A memory-off session may only delete findings it
    # authored — mirroring read_finding's wall. Gate before any read-back or
    # write.
    if not memory_enabled(conn, session_id) and owning_session_id != session_id:
        raise SkillError(
            "MEMORY_OFF",
            f"This session has memory disabled and finding {args.finding_id} "
            "was authored by another session, so it is not accessible. Start a "
            "memory-enabled session (omit --no-memory) to delete other "
            "sessions' findings.",
        )

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
