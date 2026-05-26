#!/usr/bin/env python3
"""edit_finding — update an existing finding's title, description, and/or body.

The primary use case is fixing citation formatting on a finding written by a
local model that produced ``[chunks 1, 2]`` instead of ``[^1][^2]``. The
agent re-writes the body with proper markers and calls this script; the body
is re-chunked, re-embedded, and the ``finding_citations`` rows are rebuilt.

At least one of ``--title``, ``--description``, or ``--body-file`` is required.
When ``--body-file`` is provided, the new body must contain at least one
``[^N]`` citation marker and every marker must reference a real chunk_id —
same rules as ``save_finding``.

Output mirrors ``save_finding`` (so the agent's echo-the-body contract still
works after an edit):

    {
      "finding_id": int,
      "session_id": int, "session_name": str,
      "body": str,
      "chunk_ids": [int, ...],
      "citations": [{...}, ...]
    }

``session_id`` / ``session_name`` describe the session that *owns* the
finding (its original author), not the current session doing the edit; the
audit log records the editor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bartleby.skill_runner import SkillError, run
from bartleby.skill_scripts._common import (
    extract_citations,
    rebuild_finding_chunks,
    replace_finding_citations,
    resolve_citations,
    validate_chunk_ids_exist,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="edit_finding")
    p.add_argument("--finding-id", type=int, required=True, dest="finding_id")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--description", type=str, default=None)
    p.add_argument("--body-file", type=str, default=None, dest="body_file")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def _validated_replacement(new_value: str | None, current: str, *, code: str, label: str) -> str:
    """Return ``new_value`` (validated non-blank) or ``current`` if unchanged."""
    if new_value is None:
        return current
    if not new_value.strip():
        raise SkillError(code, f"Finding {label} must be non-empty.")
    return new_value


def _load_new_body(body_file: str, conn) -> tuple[str, list[int]]:
    body_path = Path(body_file)
    if not body_path.exists() or not body_path.is_file():
        raise SkillError(
            "BODY_FILE_NOT_FOUND",
            f"--body-file path does not exist: {body_path}",
        )
    body = body_path.read_text(encoding="utf-8")
    if not body.strip():
        raise SkillError("EMPTY_BODY", "Finding body is empty.")

    citations = extract_citations(body)
    if not citations:
        raise SkillError(
            "NO_INLINE_CITATIONS",
            "Finding body must include at least one inline citation marker "
            "of the form [^<chunk_id>] (e.g. [^4192]). See SKILL.md.",
        )
    validate_chunk_ids_exist(conn, citations)
    return body, citations


def _current_chunk_and_citation_ids(cur, finding_id: int) -> tuple[list[int], list[int]]:
    chunk_ids = [
        r[0] for r in cur.execute(
            "SELECT chunk_id FROM chunks WHERE source_kind = 'finding' "
            "AND source_id = ? ORDER BY chunk_index",
            (finding_id,),
        )
    ]
    citation_ids = [
        r[0] for r in cur.execute(
            "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
            (finding_id,),
        )
    ]
    return chunk_ids, citation_ids


def work(*, conn, args, session_id) -> dict:
    if args.title is None and args.description is None and args.body_file is None:
        raise SkillError(
            "NOTHING_TO_UPDATE",
            "edit_finding requires at least one of --title, --description, "
            "or --body-file.",
        )

    cur = conn.cursor()
    existing = cur.execute(
        "SELECT session_id, title, description, body FROM findings "
        "WHERE finding_id = ?",
        (args.finding_id,),
    ).fetchone()
    if existing is None:
        raise SkillError(
            "FINDING_NOT_FOUND", f"No finding with id {args.finding_id}.",
        )
    owning_session_id, current_title, current_description, current_body = existing

    new_title = _validated_replacement(
        args.title, current_title, code="EMPTY_TITLE", label="title",
    )
    new_description = _validated_replacement(
        args.description, current_description,
        code="EMPTY_DESCRIPTION", label="description",
    )

    if args.body_file is not None:
        new_body, new_citations = _load_new_body(args.body_file, conn)
    else:
        new_body, new_citations = current_body, None

    cur.execute(
        "UPDATE findings SET title = ?, description = ?, body = ? "
        "WHERE finding_id = ?",
        (new_title, new_description, new_body, args.finding_id),
    )

    if new_citations is not None:
        chunk_ids = rebuild_finding_chunks(conn, args.finding_id, new_body)
        replace_finding_citations(conn, args.finding_id, new_citations)
        citation_ids = new_citations
    else:
        chunk_ids, citation_ids = _current_chunk_and_citation_ids(cur, args.finding_id)

    session_name = cur.execute(
        "SELECT name FROM sessions WHERE session_id = ?", (owning_session_id,)
    ).fetchone()[0]

    return {
        "finding_id": args.finding_id,
        "session_id": owning_session_id,
        "session_name": session_name,
        "body": new_body,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citation_ids),
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="edit_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
