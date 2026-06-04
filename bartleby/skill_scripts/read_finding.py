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

``FINDING_NOT_FOUND`` when the id doesn't exist. Memory-off sessions get a
``{"code": "MEMORY_OFF"}`` error envelope — findings are the agent's memory
and are inaccessible when memory is off.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    require_memory_enabled,
    resolve_citations,
    session_provenance,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_finding", __doc__)
    p.add_argument("--finding-id", type=int, required=True, dest="finding_id")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    require_memory_enabled(conn, session_id)

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

    chunk_ids = [
        r[0] for r in cur.execute(
            "SELECT chunk_id FROM chunks WHERE source_kind = 'finding' "
            "AND source_id = ? ORDER BY chunk_index",
            (args.finding_id,),
        )
    ]
    citation_ids = [
        r[0] for r in cur.execute(
            "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
            (args.finding_id,),
        )
    ]

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
