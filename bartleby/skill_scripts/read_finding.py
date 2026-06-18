#!/usr/bin/env python3
"""read_finding — read one whole finding by id.

``search --findings`` returns ranked finding *fragments*; this returns the
full finding — the same shape ``save_finding`` / ``edit_finding`` emit, so the
verbatim-body contract still holds (the agent echoes ``body`` back unchanged).

``session_id`` / ``session_name`` describe the session that *authored* the
finding, and ``model`` / ``harness`` the backend behind it (null when
unrecorded). ``chunk_ids`` are the finding's own body chunks (``source_kind =
'finding'``); ``citations`` resolve the chunks the finding cites.

``dangling_citations`` are chunk ids appearing in ``[^chunk:N]`` markers in the
body whose citation no longer resolves — the cited source has since been removed
(deleted, or rebuilt under new chunk ids by an edit/merge), and the
``ON DELETE CASCADE`` on ``finding_citations`` stripped the row. The body is
left verbatim, so the marker survives as a provenance fact: the claim *was*
supported by a source that is now gone. When compiling a report, flag or
annotate such a marker as a removed citation — don't silently drop it; that a
claim was once cited is itself signal. The cited source can't be recovered
here (only the id survives), so phrase it as "cited source no longer
available," never "deleted."

``dangling_finding_links`` are finding ids appearing in ``[^finding:N]`` markers
in the body that no longer resolve — the referenced finding has been deleted. The
marker is left verbatim (same provenance rationale as dangling chunk citations).

Output (every id is type-tagged, e.g. ``"chunk:15837"``, ``"finding:204"``):
    {
      "finding_id": "finding:<id>",
      "session_id": int, "session_name": str,
      "model": str|null, "harness": str|null,
      "title": str, "description": str,
      "body": str,
      "created_at": str,
      "chunk_ids": ["chunk:<id>", ...],
      "citations": [{
        "chunk_id": "chunk:<id>",
        "source_kind": str, "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
      }, ...],
      "external_citations": [{"scheme": "url"|"doc", "ref": str}, ...],
      "dangling_citations": ["chunk:<id>", ...],    # [^chunk:N] markers with no resolved citation
      "dangling_finding_links": ["finding:<id>", ...] # [^finding:N] markers whose target is gone
    }

``external_citations`` are the ``[^url:<url>]`` / ``[^doc:<ref>]`` markers in the
body — supplementary external attributions that ride *alongside* (never replace)
the required corpus-chunk citations. They carry no DB row; they're parsed from
the body on read, so the marker text is the single source of truth. The ``ref``
is opaque (never fetched). A finding still needs ≥1 ``[^chunk:N]`` chunk citation,
so ``external_citations`` may be present even when ``citations`` is too.

``FINDING_NOT_FOUND`` when the id doesn't exist. In a memory-off session you
can still read findings *this* session authored; reading a finding written by
another session raises ``{"code": "MEMORY_OFF"}`` (other sessions' findings
are walled off to avoid contaminating an evaluation run).
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    assert_findings_accessible,
    extract_citations,
    extract_external_citations,
    extract_finding_citations,
    finding_chunk_and_citation_ids,
    live_finding_ids,
    resolve_citations,
    session_provenance,
)
from bartleby.skill_scripts._ids import format_output_ids, prefixed_int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_finding", __doc__)
    p.add_argument(
        "--finding-id", type=prefixed_int("finding"), required=True,
        dest="finding_id", help="Type-tagged finding id, e.g. finding:204.",
    )
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
    assert_findings_accessible(conn, session_id, [args.finding_id], action="read")

    chunk_ids, citation_ids = finding_chunk_and_citation_ids(cur, args.finding_id)

    citations = resolve_citations(conn, citation_ids)
    # A [^chunk:N] marker dangles when its chunk id is absent from the *resolved*
    # citations — resolve_citations drops chunks that vanished, so the resolved
    # set is the right thing to diff against. Derived at read time; covers
    # markers orphaned by any cause (delete, edit/merge rebuild, legacy data).
    resolved_ids = {c["chunk_id"] for c in citations}
    dangling = [cid for cid in extract_citations(body) if cid not in resolved_ids]

    # A [^finding:N] marker dangles when the referenced finding no longer exists.
    # Computed at read time from the body — no DB row backs these links (#654).
    linked_ids = extract_finding_citations(body)
    alive = live_finding_ids(conn, linked_ids)
    dangling_finding_links = [fid for fid in linked_ids if fid not in alive]

    return format_output_ids({
        "finding_id": args.finding_id,
        "session_id": owning_session_id,
        **session_provenance(conn, owning_session_id),
        "title": title,
        "description": description,
        "body": body,
        "created_at": created_at,
        "chunk_ids": chunk_ids,
        "citations": citations,
        "external_citations": extract_external_citations(body),
        "dangling_citations": dangling,
        "dangling_finding_links": dangling_finding_links,
    })


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_finding", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
