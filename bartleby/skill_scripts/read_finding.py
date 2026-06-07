#!/usr/bin/env python3
"""read_finding — read one whole finding by id.

``search --findings`` returns ranked finding *fragments*; this returns the
full finding — the same shape ``save_finding`` / ``edit_finding`` emit, so the
verbatim-body contract still holds (the agent echoes ``body`` back unchanged).

``session_id`` / ``session_name`` describe the session that *authored* the
finding, and ``model`` / ``harness`` the backend behind it (null when
unrecorded). ``chunk_ids`` are the finding's own body chunks (``source_kind =
'finding'``); ``citations`` resolve the chunks the finding cites.

Output:
    {
      "finding_id": int,
      "session_id": int, "session_name": str,
      "model": str|null, "harness": str|null,
      "title": str, "description": str,
      "body": str,
      "created_at": str,
      "chunk_ids": [int, ...],
      "citations": [{
        "chunk_id": int,
        "source_kind": str, "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
      }, ...]
    }

``FINDING_NOT_FOUND`` when the id doesn't exist. In a memory-off session you
can still read findings *this* session authored; reading a finding written by
another session raises ``{"code": "MEMORY_OFF"}`` (other sessions' findings
are walled off to avoid contaminating an evaluation run).
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    finding_chunk_and_citation_ids,
    memory_enabled,
    resolve_citations,
    session_provenance,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_finding", __doc__)
    p.add_argument("--finding", type=int, required=True, dest="finding_id")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT session_id, title, description, body, created_at "
        "FROM findings WHERE finding_id = ?",
        (args.finding_id,),
    ).fetchone()
    if row is None:
        raise SkillError(
            "FINDING_NOT_FOUND", f"No finding with id {args.finding_id}.",
        )
    owning_session_id, title, description, body, created_at = row

    # Memory-off sessions can read back their *own* findings (so a run can
    # verify what it just wrote) but not another session's — that would
    # contaminate an evaluation with prior conclusions.
    if not memory_enabled(conn, session_id) and owning_session_id != session_id:
        raise SkillError(
            "MEMORY_OFF",
            f"This session has memory disabled and finding {args.finding_id} "
            "was authored by another session, so it is not accessible. Start a "
            "memory-enabled session (omit --no-memory) to read other sessions' "
            "findings.",
        )

    chunk_ids, citation_ids = finding_chunk_and_citation_ids(cur, args.finding_id)

    return {
        "finding_id": args.finding_id,
        "session_id": owning_session_id,
        **session_provenance(conn, owning_session_id),
        "title": title,
        "description": description,
        "body": body,
        "created_at": created_at,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citation_ids),
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
