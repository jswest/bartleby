#!/usr/bin/env python3
"""list_findings — enumerate findings in the corpus for browsing.

The companion to ``list_documents``, but for the agent's own memory. Findings
are *written* via ``save_finding`` / ``edit_finding`` and surfaced as ranked
fragments via ``search --findings`` — this script is the missing "what
findings exist?" path. Newest first (``ORDER BY finding_id DESC``).

Per finding: ``finding_id``, ``title``, ``description``, ``session_name``
(the session that authored it), ``created_at``, and ``citation_count`` (how
many chunks it cites). To read a whole finding, call
``read_finding --finding-id <N>``.

Output:
    {
      "findings": [{
        "finding_id": int,
        "title": str, "description": str,
        "session_name": str,
        "created_at": str,
        "citation_count": int,
      }, ...],
      "total": int,
      "offset": int, "limit": int,
      "hint": str|null         # set when more pages remain
    }

Memory-off sessions get a ``{"code": "MEMORY_OFF"}`` error envelope instead —
findings are the agent's memory and are inaccessible when memory is off.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import run
from bartleby.skill_scripts._common import require_memory_enabled


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="list_findings")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    require_memory_enabled(conn, session_id)

    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0]

    rows = cur.execute(
        "SELECT f.finding_id, f.title, f.description, s.name, f.created_at, "
        "       COALESCE(fc.n, 0) AS citation_count "
        "FROM findings f "
        "LEFT JOIN sessions s ON s.session_id = f.session_id "
        "LEFT JOIN (SELECT finding_id, COUNT(*) AS n FROM finding_citations "
        "           GROUP BY finding_id) fc ON fc.finding_id = f.finding_id "
        "ORDER BY f.finding_id DESC LIMIT ? OFFSET ?",
        (args.limit, args.offset),
    )

    findings = [
        {
            "finding_id": finding_id,
            "title": title,
            "description": description,
            "session_name": session_name,
            "created_at": created_at,
            "citation_count": citation_count,
        }
        for finding_id, title, description, session_name, created_at, citation_count in rows
    ]

    next_offset = args.offset + len(findings)
    has_more = next_offset < total
    hint = (
        f"Showing {args.offset + 1}-{next_offset} of {total}. "
        f"Pass --offset {next_offset} to continue."
        if has_more and findings else None
    )

    return {
        "findings": findings,
        "total": total,
        "offset": args.offset,
        "limit": args.limit,
        "hint": hint,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="list_findings", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
