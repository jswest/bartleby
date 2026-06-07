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
      "model": str|null, "harness": str|null,
      "body": str,
      "chunk_ids": [int, ...],
      "citations": [{...}, ...]
    }

``session_id`` / ``session_name`` (and ``model`` / ``harness``) describe the
session that *owns* the finding (its original author), not the current
session doing the edit; the audit log records the editor.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    finding_chunk_and_citation_ids,
    load_finding_body,
    rebuild_finding_chunks,
    replace_finding_citations,
    resolve_citations,
    session_provenance,
    validated_replacement,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("edit_finding", __doc__)
    p.add_argument("--finding", type=int, required=True, dest="finding_id")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--description", type=str, default=None)
    p.add_argument("--body-file", type=str, default=None, dest="body_file")
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


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

    new_title = validated_replacement(
        args.title, current_title, code="EMPTY_TITLE", label="title",
    )
    new_description = validated_replacement(
        args.description, current_description,
        code="EMPTY_DESCRIPTION", label="description",
    )

    if args.body_file is not None:
        new_body, new_citations = load_finding_body(conn, args.body_file)
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
        chunk_ids, citation_ids = finding_chunk_and_citation_ids(cur, args.finding_id)

    return {
        "finding_id": args.finding_id,
        "session_id": owning_session_id,
        **session_provenance(conn, owning_session_id),
        "body": new_body,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citation_ids),
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="edit_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
