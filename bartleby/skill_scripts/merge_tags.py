#!/usr/bin/env python3
"""merge_tags — move all assignments from one tag onto another, then delete the source.

Errors if either tag doesn't exist or if ``--from`` == ``--into``.

Output:
    {"status": "merged", "from": {"tag_id": int, "name": str},
     "into": {"tag_id": int, "name": str}, "inserted": int,
     "already_present": int,
     "value_collisions": [{"document_id": int, "kept": str, "dropped": str},
                          ...]}

``inserted`` is the number of source assignments actually copied onto the
destination tag. ``already_present`` is the number skipped because the
destination tag already carried that document (the overlap absorbed by
``INSERT OR IGNORE``). Their sum is the source tag's pre-merge assignment count.

**Same-type only.** Both tags must be the same kind — two value-tags or two
boolean tags. A mixed merge (value-tag into a boolean tag, or vice versa) is
refused with ``MIXED_TYPE_MERGE`` before anything mutates: a value-tag's
``value``/``chunk_id`` carried onto a boolean tag (``value_type IS NULL``)
would be unreachable — every value-read path gates on ``value_type IS NOT
NULL`` — and the source's ``value_type``/``pattern`` would be lost with it.

**Value collision rule (value-tags).** The merge is value-preserving on the
target's existing rows: where both tags carry a *value* for the same document,
the **target's value is kept** and the source's is dropped (reported in
``value_collisions`` with ``kept``/``dropped``, never silently lost). Where the
target carries a *plain* assignment (value NULL — someone assigned the value-tag
without extracting) and the source holds a value for that document, the source's
value (and chunk anchor) is **carried onto the target** rather than discarded —
no silent loss. For a document the source carries but the target doesn't, the
source's value (and its chunk anchor) rides along with the copied assignment.
``value_collisions`` is empty for boolean-tag merges.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._ids import format_output_ids
from bartleby.skill_scripts._tags import require_tag_by_name


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("merge_tags", __doc__)
    p.add_argument("--from", type=str, required=True, dest="from_name")
    p.add_argument("--into", type=str, required=True, dest="into_name")
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    if args.from_name == args.into_name:
        raise SkillError(
            "SELF_MERGE", "--from and --into must be different tags."
        )

    src = require_tag_by_name(conn, args.from_name)
    dst = require_tag_by_name(conn, args.into_name)

    # Same-type only: a value-tag's value/chunk_id is invisible on a boolean
    # tag (value-read paths gate on value_type IS NOT NULL) and the source's
    # value_type/pattern would vanish with the deleted source. Refuse before
    # mutating anything.
    if src.is_value_tag != dst.is_value_tag:
        src_kind = "value-tag" if src.is_value_tag else "boolean tag"
        dst_kind = "value-tag" if dst.is_value_tag else "boolean tag"
        raise SkillError(
            "MIXED_TYPE_MERGE",
            f"Cannot merge {src.name!r} ({src_kind}) into {dst.name!r} "
            f"({dst_kind}): both tags must be the same kind.",
        )

    cur = conn.cursor()
    moved_before = cur.execute(
        "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (src.tag_id,),
    ).fetchone()[0]

    # Value collisions: documents the destination already carries (so its
    # assignment + value survive) where the source also holds a distinct value.
    # Keep the target's, report the dropped source value. Computed before the
    # copy so the OR IGNORE below — which keeps the destination row untouched on
    # conflict — naturally enacts "keep target".
    collisions = [
        {"document_id": did, "kept": kept, "dropped": dropped}
        for did, kept, dropped in cur.execute(
            "SELECT s.document_id, d.value, s.value "
            "FROM document_tags s JOIN document_tags d "
            "  ON d.document_id = s.document_id AND d.tag_id = ? "
            "WHERE s.tag_id = ? AND s.value IS NOT NULL "
            "  AND d.value IS NOT NULL AND d.value IS NOT s.value",
            (dst.tag_id, src.tag_id),
        )
    ]

    # Carry the source's value/chunk onto a destination row that is a *plain*
    # assignment (value NULL) where the source holds a value. Without this the
    # OR IGNORE below would keep the destination's NULL row and the source value
    # would be deleted with the source tag — silent data loss. Done before the
    # INSERT so it only touches pre-existing destination rows.
    cur.execute(
        "UPDATE document_tags AS d SET value = s.value, chunk_id = s.chunk_id "
        "FROM document_tags AS s "
        "WHERE d.tag_id = ? AND d.value IS NULL "
        "  AND s.tag_id = ? AND s.value IS NOT NULL "
        "  AND s.document_id = d.document_id",
        (dst.tag_id, src.tag_id),
    )

    # Carry value + chunk anchor along when the destination doesn't already
    # carry the document (OR IGNORE keeps the destination's row on overlap).
    cur.execute(
        "INSERT OR IGNORE INTO document_tags "
        "(document_id, tag_id, value, chunk_id) "
        "SELECT document_id, ?, value, chunk_id "
        "FROM document_tags WHERE tag_id = ?",
        (dst.tag_id, src.tag_id),
    )
    inserted = conn.changes()
    cur.execute("DELETE FROM tags WHERE tag_id = ?", (src.tag_id,))
    return format_output_ids({
        "status": "merged",
        "from": {"tag_id": src.tag_id, "name": src.name},
        "into": {"tag_id": dst.tag_id, "name": dst.name},
        "inserted": inserted,
        "already_present": moved_before - inserted,
        "value_collisions": collisions,
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="merge_tags", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
