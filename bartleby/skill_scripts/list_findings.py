#!/usr/bin/env python3
"""list_findings — enumerate findings in the corpus for browsing.

The companion to ``list_documents``, but for the agent's own memory. Findings
are *written* via ``save_finding`` / ``edit_finding`` and surfaced as ranked
fragments via ``search --findings`` — this script is the missing "what
findings exist?" path. Newest first (``ORDER BY finding_id DESC``).

Per finding: ``finding_id``, ``title``, ``description``, ``session_name``
(the session that authored it), ``model`` / ``harness`` (the backend behind
it, null when unrecorded), ``created_at``, and ``citation_count`` (how many
chunks it cites). To read a whole finding, call
``read_finding --finding <N>``.

Output:
    {
      "findings": [{
        "finding_id": int,
        "title": str, "description": str,
        "session_name": str,
        "model": str|null, "harness": str|null,
        "created_at": str,
        "citation_count": int,
      }, ...],
      "total": int,
      "offset": int, "limit": int,
      "hint": str|null         # set when more pages remain
    }

With ``--brief`` each finding is trimmed to ``finding_id``, ``title``, and
``citation_count`` — dropping ``description``, ``session_name``,
``model``/``harness``, and ``created_at``. The envelope is unchanged.

In a memory-off session the listing is scoped to findings *this* session
authored (other sessions' findings are walled off to avoid contaminating an
evaluation run); ``total`` and pagination reflect that scoped set. A
memory-on session lists every finding.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import build_arg_parser, run
from bartleby.skill_scripts._common import (
    memory_enabled, nonneg_int, pagination_hint, positive_int,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("list_findings", __doc__)
    p.add_argument("--limit", type=positive_int, default=200)
    p.add_argument("--offset", type=nonneg_int, default=0)
    p.add_argument(
        "--brief",
        action="store_true",
        help="Skinniest tier: finding_id, title, citation_count only. Drops "
             "description, session_name, model/harness, created_at.",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    cur = conn.cursor()

    # Memory-off: own findings only (see module docstring). Memory-on: all.
    if memory_enabled(conn, session_id):
        scope_sql, scope_params = "", ()
    else:
        scope_sql, scope_params = "WHERE f.session_id = ?", (session_id,)

    total = cur.execute(
        f"SELECT COUNT(*) FROM findings f {scope_sql}", scope_params,
    ).fetchone()[0]

    rows = cur.execute(
        "SELECT f.finding_id, f.title, f.description, s.name, s.model, s.harness, "
        "       f.created_at, COALESCE(fc.n, 0) AS citation_count "
        "FROM findings f "
        "LEFT JOIN sessions s ON s.session_id = f.session_id "
        "LEFT JOIN (SELECT finding_id, COUNT(*) AS n FROM finding_citations "
        "           GROUP BY finding_id) fc ON fc.finding_id = f.finding_id "
        f"{scope_sql} "
        "ORDER BY f.finding_id DESC LIMIT ? OFFSET ?",
        (*scope_params, args.limit, args.offset),
    )

    if args.brief:
        findings = [
            {
                "finding_id": finding_id,
                "title": title,
                "citation_count": citation_count,
            }
            for finding_id, title, description, session_name, model, harness, created_at, citation_count in rows
        ]
    else:
        findings = [
            {
                "finding_id": finding_id,
                "title": title,
                "description": description,
                "session_name": session_name,
                "model": model,
                "harness": harness,
                "created_at": created_at,
                "citation_count": citation_count,
            }
            for finding_id, title, description, session_name, model, harness, created_at, citation_count in rows
        ]

    hint = pagination_hint(args.offset, len(findings), total)

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
