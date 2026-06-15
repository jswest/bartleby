#!/usr/bin/env python3
"""extract — run a value-tag's stored regex over a set of chunks, storing the
per-document values it captures.

A value-tag (created with ``add_tag --value-type/--pattern``) carries a
deterministic extraction regex. ``extract`` applies that regex to an explicit,
agent-supplied set of chunk ids — there is no implicit corpus-wide sweep: you
have already located the chunks (via ``search``/``scan``), so you pass them.

  extract --tag revenue --chunks chunk:4192,chunk:4193,chunk:4201

Per chunk: the tag's ``pattern`` is run (linear-time, via re2) over the chunk
text; on a match the ``(?P<value>…)`` capture group is cast per the tag's
``value_type`` (number/string/date) and written to ``document_tags.value`` +
``.chunk_id`` for that chunk's document. Batch many chunk ids in one call so
interpreter startup is paid once.

Honesty disciplines (mirroring the tag classifier's ``failed`` list):
  - **No match → no value, never fabricated.** Chunks whose text the pattern
    doesn't match land in ``no_match`` and are skipped; the rest proceed.
  - **One value per document.** If two *distinct* values are captured for the
    same document (across the passed chunks), that's an ambiguity: it's reported
    in ``conflicts`` and **neither** is stored — the script never silently
    picks. (Two chunks yielding the *same* value are not a conflict; the first
    chunk's match is stored as the anchor.)
  - **Bad cast → no value.** A match whose captured span can't satisfy the
    value_type (e.g. a "number" tag capturing non-numeric text) lands in
    ``cast_errors`` and is skipped.

Re-runnable: as the corpus grows, re-locate chunks and re-run the same stored
pattern over the new ids. A stored value is overwritten by a later
unambiguous extraction for the same (tag, document).

Output (every id is type-tagged, e.g. ``"chunk:4192"``, ``"document:3"``,
``"tag:7"``):
    {
      "tag": {"tag_id": "tag:<id>", "name": str, "value_type": str, "pattern": str},
      "stored": [{"document_id": "document:<id>", "value": str, "chunk_id": "chunk:<id>"}, ...],
      "conflicts": [{"document_id": "document:<id>",
                     "values": [{"value": str, "chunk_id": "chunk:<id>"}, ...]}, ...],
      "no_match": ["chunk:<id>", ...],          # chunk ids the pattern didn't match
      "cast_errors": [{"chunk_id": "chunk:<id>", "captured": str, "error": str}, ...],
      "missing": ["chunk:<id>", ...]            # chunk ids that don't exist
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._ids import (
    format_id, format_output_ids, prefixed_int_list,
)
from bartleby.skill_scripts._tags import (
    cast_value,
    compile_pattern,
    require_tag_by_name,
    require_value_tag,
    upsert_value,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("extract", __doc__)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument(
        "--chunks", type=prefixed_int_list("chunk"), required=True,
        dest="chunk_ids",
        help="Comma-separated type-tagged chunk ids, e.g. chunk:4192,chunk:4193.",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    tag = require_tag_by_name(conn, args.tag)
    require_value_tag(tag)
    compiled = compile_pattern(tag.pattern)

    # Resolve chunk text + owning document in one pass; only 'document' chunks
    # carry a document_id (source_id), which is the anchor for a stored value.
    ordered = list(dict.fromkeys(args.chunk_ids))  # dedup, preserve order
    ph = ",".join("?" * len(ordered))
    rows = {
        cid: (source_kind, source_id, text)
        for cid, source_kind, source_id, text in conn.cursor().execute(
            f"SELECT chunk_id, source_kind, source_id, text FROM chunks "
            f"WHERE chunk_id IN ({ph})",
            ordered,
        )
    }
    missing = [cid for cid in ordered if cid not in rows]

    no_match: list[int] = []
    cast_errors: list[dict] = []
    # document_id -> {value: first chunk_id seen with that value} (insertion
    # order preserved so the first matching chunk is the stored anchor).
    per_doc: dict[int, dict[str, int]] = {}

    for cid in ordered:
        row = rows.get(cid)
        if row is None:
            continue
        source_kind, source_id, text = row
        if source_kind != "document":
            # Non-document chunks (summary/finding/image) have no owning
            # document to anchor a value to — treat as a non-match for honesty.
            no_match.append(cid)
            continue
        m = compiled.search(text)
        if m is None:
            no_match.append(cid)
            continue
        captured = m.group("value")
        try:
            value = cast_value(tag.value_type, captured)
        except SkillError as e:
            cast_errors.append(
                {"chunk_id": cid, "captured": captured, "error": e.message}
            )
            continue
        per_doc.setdefault(source_id, {}).setdefault(value, cid)

    stored: list[dict] = []
    conflicts: list[dict] = []
    for document_id, value_to_chunk in per_doc.items():
        if len(value_to_chunk) > 1:
            # Distinct values for one document — ambiguity. Store neither.
            conflicts.append({
                "document_id": document_id,
                "values": [
                    {"value": v, "chunk_id": c}
                    for v, c in value_to_chunk.items()
                ],
            })
            continue
        (value, chunk_id), = value_to_chunk.items()
        upsert_value(conn, document_id, tag.tag_id, value, chunk_id)
        stored.append(
            {"document_id": document_id, "value": value, "chunk_id": chunk_id}
        )

    return format_output_ids({
        "tag": {
            "tag_id": tag.tag_id, "name": tag.name,
            "value_type": tag.value_type, "pattern": tag.pattern,
        },
        "stored": stored,
        "conflicts": conflicts,
        # no_match / missing are chunk-id lists (not in the field map): tag each.
        "no_match": [format_id("chunk", cid) for cid in no_match],
        "cast_errors": cast_errors,
        "missing": [format_id("chunk", cid) for cid in missing],
    })


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="extract", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
