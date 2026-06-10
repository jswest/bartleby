#!/usr/bin/env python3
"""merge_findings — fold N findings into one, then delete the sources.

The after-the-fact cleanup for finding clusters that should have been one
finding (overlapping report iterations, redundant drafts). One existing
finding is kept as the ``--into`` target so its ``finding_id`` / provenance
survives; the agent supplies the consolidated markdown via ``--body-file``
(same citation rules as ``save_finding`` — at least one ``[^N]`` marker, every
marker a real chunk_id), the target's body is replaced and its citations
re-extracted, and the ``--from`` sources are deleted (body chunks +
``finding_citations`` cleared, cited *document* chunks untouched).

``--title`` / ``--description`` are optional; omit them to keep the target's
current values. The target must not appear in ``--from``.

Output mirrors ``save_finding`` (so the verbatim-body echo contract holds)
plus ``merged_from``:

    {
      "finding_id": int,            # the surviving target
      "session_id": int,            # the target's author
      "session_name": str, "model": str|null, "harness": str|null,
      "body": str,
      "chunk_ids": [int, ...],
      "citations": [{
        "chunk_id": int,
        "source_kind": str, "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
      }, ...],
      "merged_from": [int, ...]     # source ids folded in and deleted
    }

``FINDING_NOT_FOUND`` (with the offending ids) when the target or any source
is missing; ``TARGET_IN_SOURCES`` when ``--into`` is also in ``--from``. In a
memory-off session every finding involved (the ``--into`` target and all
``--from`` sources) must have been authored by *this* session; any finding
written by another session raises ``{"code": "MEMORY_OFF"}`` (with the foreign
ids) — merging consumes other sessions' findings, walled off to avoid
contaminating an evaluation run.
"""

from __future__ import annotations

import argparse

from bartleby.db.chunks import delete_chunks_for
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    assert_findings_accessible,
    comma_int_list,
    load_finding_body,
    rebuild_finding_chunks,
    reject_citations_to_involved_findings,
    replace_finding_citations,
    resolve_citations,
    session_provenance,
    validated_replacement,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("merge_findings", __doc__)
    p.add_argument(
        "--from", type=comma_int_list("finding_id"), required=True,
        dest="from_ids",
    )
    p.add_argument("--into", type=int, required=True, dest="into")
    p.add_argument("--body-file", type=str, required=True, dest="body_file")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--description", type=str, default=None)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    target = args.into
    # Dedup sources, preserving first-appearance order.
    sources = list(dict.fromkeys(args.from_ids))

    if target in sources:
        raise SkillError(
            "TARGET_IN_SOURCES",
            f"--into {target} cannot also appear in --from.",
        )

    cur = conn.cursor()
    ids = sources + [target]
    ph = ",".join("?" * len(ids))
    owners = {
        r[0]: r[1] for r in cur.execute(
            f"SELECT finding_id, session_id FROM findings "
            f"WHERE finding_id IN ({ph})",
            ids,
        )
    }
    missing = [fid for fid in ids if fid not in owners]
    if missing:
        raise SkillError(
            "FINDING_NOT_FOUND",
            f"No finding(s) with id(s): {missing}.",
            missing_finding_ids=missing,
        )

    # Merging deletes the sources and folds them into the target. A memory-off
    # session may only touch findings it authored — mirroring read_finding's
    # wall, across every finding involved. Gate before any write.
    assert_findings_accessible(conn, session_id, ids, action="merge")

    body, citations = load_finding_body(conn, args.body_file)

    # The merge deletes-and-rebuilds the body chunks of the target *and* the
    # sources. A citation pointing at any of those chunks would either be
    # cascade-deleted into a dangling [^N] (sources) or hit an FK violation
    # surfacing as a bare INTERNAL_ERROR (target rebuilt before its citations
    # are replaced). Reject upfront, naming the offending chunk ids.
    reject_citations_to_involved_findings(
        conn, citations, ids, code="CITES_MERGED_CHUNKS", action="merge",
    )

    owning_session_id = owners[target]
    current_title, current_description = cur.execute(
        "SELECT title, description FROM findings WHERE finding_id = ?",
        (target,),
    ).fetchone()
    new_title = validated_replacement(
        args.title, current_title, code="EMPTY_TITLE", label="title",
    )
    new_description = validated_replacement(
        args.description, current_description,
        code="EMPTY_DESCRIPTION", label="description",
    )

    cur.execute(
        "UPDATE findings SET title = ?, description = ?, body = ? "
        "WHERE finding_id = ?",
        (new_title, new_description, body, target),
    )
    chunk_ids = rebuild_finding_chunks(conn, target, body)
    replace_finding_citations(conn, target, citations)
    for src in sources:
        delete_chunks_for(conn, "finding", src)
    src_ph = ",".join("?" * len(sources))
    cur.execute(
        f"DELETE FROM findings WHERE finding_id IN ({src_ph})", sources,
    )

    return {
        "finding_id": target,
        "session_id": owning_session_id,
        **session_provenance(conn, owning_session_id),
        "body": body,
        "chunk_ids": chunk_ids,
        "citations": resolve_citations(conn, citations),
        "merged_from": sources,
    }


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="merge_findings", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
