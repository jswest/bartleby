#!/usr/bin/env python3
"""save_finding — persist a finding (markdown with inline `[^chunk_id]` citations).

The body comes from ``--body-file`` (not ``--body``) because findings can be
long markdown and shell-escaping them is a nightmare.

Citations are derived from the body itself: every ``[^N]`` marker in the
prose is a citation, where ``N`` is a chunk_id. The body is the single source
of truth — there is no separate ``--citations`` argument.

Output:
    {
      "finding_id": int,
      "session_id": int, "session_name": str,
      "body": str,
      "chunk_ids": [int, ...],
      "citations": [{
        "chunk_id": int,
        "source_kind": str, "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
      }, ...]
    }

``body`` is the exact markdown that landed in ``findings.body``. The agent
is expected to echo it verbatim back to the user — see SKILL.md for the
single-source-of-truth contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    extract_citations,
    rebuild_finding_chunks,
    reject_malformed_citations,
    replace_finding_citations,
    resolve_citations,
    validate_chunk_ids_exist,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("save_finding", __doc__)
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--description", type=str, required=True)
    p.add_argument("--body-file", type=str, required=True, dest="body_file")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def _read_body(body_file: str) -> str:
    body_path = Path(body_file)
    if not body_path.exists() or not body_path.is_file():
        raise SkillError(
            "BODY_FILE_NOT_FOUND",
            f"--body-file path does not exist: {body_path}",
        )
    body = body_path.read_text(encoding="utf-8")
    if not body.strip():
        raise SkillError("EMPTY_BODY", "Finding body is empty.")
    return body


def _citations_from_body(conn, body: str) -> list[int]:
    """Extract markers, require at least one, and verify each chunk exists."""
    reject_malformed_citations(body)
    citations = extract_citations(body)
    if not citations:
        raise SkillError(
            "NO_INLINE_CITATIONS",
            "Finding body must include at least one inline citation marker "
            "of the form [^<chunk_id>] (e.g. [^4192]). See SKILL.md.",
        )
    validate_chunk_ids_exist(conn, citations)
    return citations


def work(*, conn, args, session_id) -> dict:
    if not args.title or not args.title.strip():
        raise SkillError("EMPTY_TITLE", "Finding title must be non-empty.")
    if not args.description or not args.description.strip():
        raise SkillError(
            "EMPTY_DESCRIPTION", "Finding description must be non-empty."
        )

    body = _read_body(args.body_file)
    citations = _citations_from_body(conn, body)

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, args.title, args.description, body),
    )
    finding_id = conn.last_insert_rowid()

    chunk_ids = rebuild_finding_chunks(conn, finding_id, body)
    replace_finding_citations(conn, finding_id, citations)

    session_name = cur.execute(
        "SELECT name FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0]

    return {
        "finding_id": finding_id,
        "session_id": session_id,
        "session_name": session_name,
        "body": body,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citations),
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="save_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
