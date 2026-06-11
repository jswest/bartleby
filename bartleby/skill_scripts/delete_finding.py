#!/usr/bin/env python3
"""delete_finding — retract a finding from the corpus.

The missing curation primitive: ``save_finding`` / ``edit_finding`` only ever
add or revise, so stale duplicates (zero-citation drafts, superseded
iterations) accumulate with no way out. This removes one finding outright.

Deletion touches the finding's own rows — its body chunks (``source_kind =
'finding'``, cleared from ``chunks`` / ``chunks_fts`` / ``chunks_vec`` via
``delete_chunks_for``) and its outgoing ``finding_citations`` (which cascade
when the ``findings`` row is deleted). The cited *document* chunks are
untouched: findings are derivative hints, never evidence.

There *is* referential fallout in one case: a finding's body chunks are
themselves citable, so another finding may have cited *this* one. When this
finding's chunks are deleted, ``finding_citations.chunk_id ... ON DELETE
CASCADE`` strips those incoming citation rows too — leaving the other
finding's body with a now-unresolved ``[^N]`` marker. That dangling marker is
surfaced at the read seam (``read_finding``'s ``dangling_citations`` and the
web finding view), not counted here.

Output:
    {
      "status": "deleted",
      "finding_id": int,
      "title": str,
      "removed_chunks": int,        # finding body chunks removed
      # this finding's OWN outgoing finding_citations rows removed — NOT the
      # incoming citations the cascade severs in other findings.
      "removed_citations": int
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
from bartleby.skill_scripts._common import assert_findings_accessible, positive_int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("delete_finding", __doc__)
    p.add_argument("--finding", type=positive_int, required=True, dest="finding_id")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT title FROM findings WHERE finding_id = ?",
        (args.finding_id,),
    ).fetchone()
    if row is None:
        raise SkillError(
            "FINDING_NOT_FOUND", f"No finding with id {args.finding_id}.",
        )
    (title,) = row

    # The response echoes the deleted title, and the deletion mutates the
    # finding outright. A memory-off session may only delete findings it
    # authored — mirroring read_finding's wall. Gate before any read-back or
    # write.
    assert_findings_accessible(conn, session_id, [args.finding_id], action="delete")

    n_citations = cur.execute(
        "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
        (args.finding_id,),
    ).fetchone()[0]

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
    run(
        tool_name="delete_finding", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
