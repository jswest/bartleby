#!/usr/bin/env python3
"""edit_finding — update an existing finding's title, description, and/or body.

The primary use case is fixing citation formatting on a finding written by a
local model that produced ``[chunks 1, 2]`` instead of ``[^chunk:1][^chunk:2]``.
The agent re-writes the body with proper markers and calls this script; the body
is re-chunked, re-embedded, and the ``finding_citations`` rows are rebuilt.

At least one of ``--title``, ``--description``, or ``--body-file`` is required.
When ``--body-file`` is provided, the new body must contain at least one
``[^chunk:N]`` citation marker and every marker must reference a real chunk_id —
same rules as ``save_finding``.

Output mirrors ``save_finding`` (every id type-tagged; the echo-the-body
contract still works after an edit):

    {
      "finding_id": "finding:<id>",
      "session_id": int, "session_name": str,
      "model": str|null, "harness": str|null,
      "body": str,
      "chunk_ids": ["chunk:<id>", ...],
      "citations": [{...}, ...],
      "external_citations": [{"scheme": "url"|"doc", "ref": str}, ...]
    }

``external_citations`` echo the ``[^url:<url>]`` / ``[^doc:<ref>]`` markers in
the body — supplementary external attributions alongside (never replacing) the
required ``[^chunk:N]`` chunk citations; no DB row, parsed from the body, ref opaque.

``session_id`` / ``session_name`` (and ``model`` / ``harness``) describe the
session that *owns* the finding (its original author), not the current
session doing the edit; the audit log records the editor.

In a memory-off session you can still edit findings *this* session authored;
editing one written by another session raises ``{"code": "MEMORY_OFF"}``. The
response echoes the body, so an ungated edit would leak another session's
finding — the same wall ``read_finding`` enforces.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    assert_findings_accessible,
    embed_body_chunks,
    extract_external_citations,
    finding_chunk_and_citation_ids,
    load_finding_body,
    reject_citations_to_involved_findings,
    replace_finding_citations,
    resolve_citations,
    session_provenance,
    validated_replacement,
    write_finding_chunks,
)
from bartleby.skill_scripts._ids import format_output_ids, prefixed_int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("edit_finding", __doc__)
    p.add_argument(
        "--finding-id", type=prefixed_int("finding"), required=True,
        dest="finding_id", help="Type-tagged finding id, e.g. finding:204.",
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--description", type=str, default=None)
    p.add_argument("--body-file", type=str, default=None, dest="body_file")
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

    # The response echoes the body, so editing a finding owned by another
    # session would leak its content (and mutate it as a side effect). A
    # memory-off session may only touch findings it authored — mirroring
    # read_finding's wall. Gate before any validation or write.
    assert_findings_accessible(conn, session_id, [args.finding_id], action="edit")

    new_title = validated_replacement(
        args.title, current_title, code="EMPTY_TITLE", label="title",
    )
    new_description = validated_replacement(
        args.description, current_description,
        code="EMPTY_DESCRIPTION", label="description",
    )

    if args.body_file is not None:
        new_body, new_citations = load_finding_body(conn, args.body_file)
        # The edit deletes-and-rebuilds this finding's own body chunks. A new
        # body citing one of those pre-edit chunk ids would hit the FK path
        # (chunks deleted before citations are replaced) and surface as a bare
        # INTERNAL_ERROR. Reject upfront, naming the offending chunk ids.
        reject_citations_to_involved_findings(
            conn, new_citations, [args.finding_id],
            code="CITES_OWN_CHUNKS", action="edit",
        )
        # Embed BEFORE the UPDATE (the first write) so the transaction's write
        # lock doesn't span the lazy model load (issue #340).
        chunk_inputs = embed_body_chunks(new_body)
    else:
        new_body, new_citations = current_body, None

    cur.execute(
        "UPDATE findings SET title = ?, description = ?, body = ? "
        "WHERE finding_id = ?",
        (new_title, new_description, new_body, args.finding_id),
    )

    if new_citations is not None:
        chunk_ids = write_finding_chunks(conn, args.finding_id, chunk_inputs)
        replace_finding_citations(conn, args.finding_id, new_citations)
        citation_ids = new_citations
    else:
        chunk_ids, citation_ids = finding_chunk_and_citation_ids(cur, args.finding_id)

    return format_output_ids({
        "finding_id": args.finding_id,
        "session_id": owning_session_id,
        **session_provenance(conn, owning_session_id),
        "body": new_body,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citation_ids),
        "external_citations": extract_external_citations(new_body),
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="edit_finding", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
