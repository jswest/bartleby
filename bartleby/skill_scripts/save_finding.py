#!/usr/bin/env python3
"""save_finding — persist a finding (markdown with inline `[^chunk:<id>]` citations).

The body comes from ``--body-file`` (not ``--body``) because findings can be
long markdown and shell-escaping them is a nightmare.  For the same reason,
``--title`` / ``--description`` have file-based siblings: ``--title-file`` /
``--description-file`` read the value verbatim from a file path, bypassing all
shell expansion.  Pass exactly one of ``--title`` or ``--title-file`` (mutually
exclusive, both required to pick one); same for ``--description`` /
``--description-file``.

Citations are derived from the body itself: every ``[^chunk:N]`` marker in the
prose is a citation, where ``N`` is a chunk_id you were handed this session. The
body is the single source of truth — there is no separate ``--citations``
argument. Untyped (``[N]`` / ``[^N]``) and wrong-type (``[^document:N]`` /
``[^finding:N]``) markers are rejected.

Output (every id is type-tagged, e.g. ``"chunk:15837"``, ``"finding:204"``):
    {
      "finding_id": "finding:<id>",
      "session_id": int, "session_name": str,
      "model": str|null, "harness": str|null,
      "title": str,
      "description": str,
      "body": str,
      "chunk_ids": ["chunk:<id>", ...],
      "citations": [{
        "chunk_id": "chunk:<id>",
        "source_kind": str, "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
      }, ...],
      "external_citations": [{"scheme": "url"|"doc", "ref": str}, ...]
    }

``external_citations`` echo the ``[^url:<url>]`` / ``[^doc:<ref>]`` markers in
the body — supplementary external attributions that ride alongside (never
replace) the required ``[^chunk:N]`` chunk citations. They carry no DB row; the
body text is their source of truth. The ref is opaque (never fetched).

``body`` is the exact markdown that landed in ``findings.body``. The agent
is expected to echo it verbatim back to the user — see SKILL.md for the
single-source-of-truth contract.

``model`` / ``harness`` describe the backend behind the authoring session
(issue #62); both are null when the backend wasn't recorded.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    embed_body_chunks,
    extract_external_citations,
    load_finding_body,
    read_text_arg,
    replace_finding_citations,
    resolve_citations,
    session_provenance,
    write_finding_chunks,
)
from bartleby.skill_scripts._ids import format_output_ids


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("save_finding", __doc__)
    title_g = p.add_mutually_exclusive_group(required=True)
    title_g.add_argument("--title", type=str)
    title_g.add_argument("--title-file", dest="title_file")
    desc_g = p.add_mutually_exclusive_group(required=True)
    desc_g.add_argument("--description", type=str)
    desc_g.add_argument("--description-file", dest="description_file")
    p.add_argument("--body-file", type=str, required=True, dest="body_file")
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    title = read_text_arg(
        args.title, args.title_file, flag="title", error_code="TITLE_FILE_NOT_FOUND",
    )
    description = read_text_arg(
        args.description, args.description_file,
        flag="description", error_code="DESCRIPTION_FILE_NOT_FOUND",
    )
    if not title or not title.strip():
        raise SkillError("EMPTY_TITLE", "Finding title must be non-empty.")
    if not description or not description.strip():
        raise SkillError(
            "EMPTY_DESCRIPTION", "Finding description must be non-empty."
        )

    body, citations = load_finding_body(conn, args.body_file)

    # Embed BEFORE the first write so the transaction's write lock doesn't span
    # the lazy model load (issue #340 — apsw's txn is deferred, so no lock is
    # held until the INSERT below).
    chunk_inputs = embed_body_chunks(body)

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, title, description, body),
    )
    finding_id = conn.last_insert_rowid()

    chunk_ids = write_finding_chunks(conn, finding_id, chunk_inputs)
    replace_finding_citations(conn, finding_id, citations)

    return format_output_ids({
        "finding_id": finding_id,
        "session_id": session_id,
        **session_provenance(conn, session_id),
        "title": title,
        "description": description,
        "body": body,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citations),
        "external_citations": extract_external_citations(body),
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="save_finding", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
