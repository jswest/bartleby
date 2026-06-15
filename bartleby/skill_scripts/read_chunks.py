#!/usr/bin/env python3
"""read_chunks — read chunks by document (paginated), by chunk_id list, or
around a target chunk.

All ids are type-tagged on both input and output (e.g. ``chunk:4192``,
``document:204``).

Three modes (mutually exclusive):

  read_chunks --document-id document:<id> [--offset N] [--limit N]
      Paginated read of a single document's chunks in chunk_index order.
      Output includes a ``document`` field and pagination metadata.

  read_chunks --chunks chunk:4192,chunk:4193,chunk:4194
      Direct lookup by chunk_id. Returns those chunks regardless of source.
      Each chunk carries its source_kind/source_id/chunk_index so the agent
      can locate it. Output includes a ``requested`` and ``missing`` list.

  read_chunks --around-chunk chunk:<id> [--window N]
      Neighborhood read: returns the target chunk plus N chunks on each
      side (default ``--window 3``), in chunk_index order. Source is
      derived from the target chunk — no need to pass --document-id. Works
      for any source kind, though image chunks have no neighbors.

The modes are mutually exclusive: pick one, and any flags belonging to the
other modes are silently ignored (e.g. ``--window`` is read only in
``--around-chunk`` mode, ``--offset``/``--limit`` only in ``--document-id``
mode).

In a memory-off session the finding wall (see ``read_finding``) extends here:
finding-kind chunks authored by *another* session are walled off so an
evaluation run can't read prior conclusions by chunk_id. ``--chunks`` drops
such ids into ``missing`` (no text or source_name is returned); an
``--around-chunk`` whose target is a foreign finding chunk raises
``{"code": "MEMORY_OFF"}``. A session's own findings and all
document/summary/image chunks are unaffected.

All modes accept ``--preview N`` to truncate each chunk's ``text`` to the
first ``N`` characters (followed by ``…`` when truncation occurred). Useful
for structural scans when you don't need full prose. Omit ``--preview`` to
get full text. Every returned chunk always carries ``text_length`` — the
pre-truncation length of the chunk's text — so the agent can size-budget
and tell which chunks were trimmed.

All modes also accept ``--returning <field>,...`` to project each chunk to
exactly the named fields, in the order given (the envelope — ``mode`` /
``document`` / ``target`` / ``missing`` / pagination — is untouched).
Selectable fields: ``chunk_id``, ``document_id``, ``source_kind``,
``source_id``, ``source_name``, ``file_name``, ``page_number``,
``chunk_index``, ``section_heading``, ``content_type``, ``text``,
``text_length``. ``document_id`` is the originating document for a
document-kind chunk and ``null`` otherwise. ``--document-id`` / ``--around-chunk``
default rows omit the ``source_*`` / ``file_name`` columns (the envelope names
the document/target once), but ``--returning`` can pull them in any mode. An
unknown field returns an ``UNKNOWN_RETURNING_FIELD`` error naming the valid set.

Paginated output:
    {
      "mode": "document",
      "document": {"id": "document:<id>", "file_name": str},
      "offset": int, "limit": int, "total": int,
      "preview": int|null,
      "chunks": [{
        "chunk_id": "chunk:<id>", "chunk_index": int,
        "section_heading": str|null,
        "page_number": int|null,
        "content_type": str|null,
        "text": str,
        "text_length": int,
      }, ...]
    }

Direct-lookup output:
    {
      "mode": "chunks",
      "requested": ["chunk:<id>", ...],
      "missing": ["chunk:<id>", ...],
      "hints": {"chunk:<id>": str, ...},   # present only when a missing id is a live document_id
      "preview": int|null,
      "chunks": [{
        "chunk_id": "chunk:<id>",
        "source_kind": str, "source_id": "<source_kind>:<id>", "source_name": str,
        "file_name": str|null,
        "page_number": int|null,
        "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
        "text_length": int,
      }, ...]
    }

Around-chunk output:
    {
      "mode": "around",
      "target": {"chunk_id": "chunk:<id>", "chunk_index": int,
                 "source_kind": str, "source_id": "<source_kind>:<id>", "source_name": str},
      "window": int,
      "preview": int|null,
      "chunks": [{
        "chunk_id": "chunk:<id>", "chunk_index": int,
        "section_heading": str|null,
        "page_number": int|null,
        "content_type": str|null,
        "text": str,
        "text_length": int,
      }, ...]
    }
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    add_returning_arg, apply_preview, assert_findings_accessible,
    chunk_locations, memory_enabled, nonneg_int,
    owned_finding_ids, positive_int, project_row, source_names,
    validate_returning,
)
from bartleby.skill_scripts._ids import (
    format_id, format_output_ids, format_source_id, prefixed_int,
    prefixed_int_list,
)


# --returning whitelist for every chunk row read_chunks emits, across all three
# modes. chunk_id + document_id lead so a citable id is always selectable.
# document_id is the originating document for a document-kind chunk and null
# otherwise (summaries/images/findings have no single document anchor here) —
# honest-null, never faked. The --document-id and --around-chunk default rows are a
# locator-light subset (no source_* / file_name), but --returning can pull the
# full set in any mode since every row carries it underneath.
CHUNK_FIELDS = [
    "chunk_id", "document_id", "source_kind", "source_id", "source_name",
    "file_name", "page_number", "chunk_index", "section_heading",
    "content_type", "text", "text_length",
]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("read_chunks", __doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--document-id", type=prefixed_int("document"), dest="document_id",
        help="Type-tagged document id, e.g. document:204.",
    )
    mode.add_argument(
        "--chunks",
        type=prefixed_int_list("chunk"),
        dest="chunk_ids",
        help="Comma-separated type-tagged chunk ids, e.g. chunk:4192,chunk:4193.",
    )
    mode.add_argument(
        "--around-chunk",
        type=prefixed_int("chunk"),
        dest="around_chunk",
        help="Target chunk id (e.g. chunk:4192); returns target plus --window "
             "chunks on each side.",
    )
    p.add_argument("--offset", type=nonneg_int, default=0)
    p.add_argument("--limit", type=positive_int, default=50)
    p.add_argument(
        "--window",
        type=nonneg_int,
        default=3,
        help="Used with --around-chunk: neighbors on each side (default 3).",
    )
    p.add_argument(
        "--preview",
        type=positive_int,
        default=None,
        help="Truncate each chunk's text to the first N chars (append '…' if trimmed).",
    )
    add_returning_arg(p, CHUNK_FIELDS)
    return p.parse_args(argv)


def _chunks_from_rows(
    rows, preview: int | None, *, returning=None, source=None,
) -> list[dict]:
    """Build the per-chunk dicts for --document-id / --around-chunk modes.

    Without ``--returning`` each row is the locator-light default (no source_*/
    file_name — both modes already name the document/target once at the
    envelope level). With ``--returning``, ``source`` supplies the shared
    per-mode context — ``{source_kind, source_id, source_name, file_name,
    document_id}`` — to complete the full whitelisted row before projecting.
    """
    out = []
    for cid, idx, heading, page, ctype, text in rows:
        default = {
            "chunk_id": cid,
            "chunk_index": idx,
            "section_heading": heading,
            "page_number": page,
            "content_type": ctype,
            "text": apply_preview(text, preview),
            "text_length": len(text),
        }
        if returning is None:
            out.append(default)
            continue
        full = {**default, **source}
        out.append(project_row(full, returning, CHUNK_FIELDS))
    return out


def _read_by_chunk_ids(
    conn, chunk_ids: list[int], preview: int | None, *, mem: bool, session_id: int,
    returning=None,
) -> dict:
    ordered = list(dict.fromkeys(chunk_ids))  # dedup, preserve order

    placeholders = ",".join("?" * len(ordered))
    rows = {
        row[0]: row
        for row in conn.cursor().execute(
            f"SELECT chunk_id, source_kind, source_id, chunk_index, "
            f"       section_heading, content_type, text "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            ordered,
        )
    }

    # Memory wall: a memory-off session never sees another session's finding
    # chunks. Drop them before any text/source_name is read, so they fall into
    # ``missing`` as if they didn't exist (mirrors read_finding's MEMORY_OFF).
    if not mem:
        finding_sids = {r[2] for r in rows.values() if r[1] == "finding"}
        owned = owned_finding_ids(conn, finding_sids, session_id)
        rows = {
            cid: r
            for cid, r in rows.items()
            if not (r[1] == "finding" and r[2] not in owned)
        }

    names = source_names(conn, {(r[1], r[2]) for r in rows.values()})
    locations = chunk_locations(conn, list(rows.keys()))

    missing = [cid for cid in ordered if cid not in rows]
    chunks = []
    for cid in ordered:
        if cid not in rows:
            continue
        _, sk, sid, chunk_index, section_heading, content_type, text = rows[cid]
        loc = locations.get(cid, {"file_name": None, "page_number": None})
        full = {
            "chunk_id": cid,
            "document_id": sid if sk == "document" else None,
            "source_kind": sk,
            # source_id is polymorphic: prefix by source_kind, not the map.
            "source_id": format_source_id(sk, sid),
            "source_name": names.get((sk, sid), ""),
            "file_name": loc["file_name"],
            "page_number": loc["page_number"],
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "content_type": content_type,
            "text": apply_preview(text, preview),
            "text_length": len(text),
        }
        projected = project_row(full, returning, CHUNK_FIELDS)
        if projected is not None:
            chunks.append(projected)
        else:
            # Default --chunks row keeps the pre-#419 shape (document_id is
            # selectable via --returning but not in the default contract).
            chunks.append({k: full[k] for k in (
                "chunk_id", "source_kind", "source_id", "source_name",
                "file_name", "page_number", "chunk_index", "section_heading",
                "content_type", "text", "text_length",
            )})

    # Cross-namespace hint: a missing chunk_id that exists as a document_id is
    # very likely a document id passed to --chunks. Surface a per-id hint so a
    # silently-wrong-namespace lookup gets caught before it becomes a citation.
    # Memory-walled foreign finding chunks also land in ``missing``, but their
    # ids ARE chunk ids (chunk and document ids overlap freely), so a walled id
    # that happens to collide with a live document_id would otherwise mis-fire
    # this hint. Suppress the hint for any missing id that still exists as a
    # chunk_id — the id is genuinely a chunk in this corpus, just walled, so the
    # director's rule (never hint on an id valid in the requested namespace)
    # holds. The hint fires only for ids that are not chunks here at all.
    live_chunk_ids = _live_chunk_ids(conn, missing)
    doc_ids = _live_document_ids(conn, missing)
    # Key the hint by the type-tagged chunk id the agent passed; the body now
    # points at the typed --document-id form (a document:<id>, not a bare int).
    hints = {
        format_id("chunk", cid): (
            f"chunk:{cid} is a document_id — did you mean "
            f"--document-id document:{cid}?"
        )
        for cid in missing
        if cid in doc_ids and cid not in live_chunk_ids
    }

    out = {
        "mode": "chunks",
        # requested/missing are chunk ids; type-tag them (not in the field map).
        "requested": [format_id("chunk", cid) for cid in ordered],
        "missing": [format_id("chunk", cid) for cid in missing],
        "preview": preview,
        "chunks": chunks,
    }
    if hints:
        out["hints"] = hints
    return format_output_ids(out)


def _live_chunk_ids(conn, ids: list[int]) -> set[int]:
    """Subset of ``ids`` that exist as ``chunk_id``s (any source kind)."""
    if not ids:
        return set()
    ph = ",".join("?" * len(ids))
    return {
        row[0]
        for row in conn.cursor().execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({ph})", ids,
        )
    }


def _live_document_ids(conn, ids: list[int]) -> set[int]:
    """Subset of ``ids`` that exist as ``document_id``s."""
    if not ids:
        return set()
    ph = ",".join("?" * len(ids))
    return {
        row[0]
        for row in conn.cursor().execute(
            f"SELECT document_id FROM documents WHERE document_id IN ({ph})", ids,
        )
    }


def _read_by_document(conn, args) -> dict:
    cur = conn.cursor()
    doc_row = cur.execute(
        "SELECT document_id, file_name FROM documents WHERE document_id = ?",
        (args.document_id,),
    ).fetchone()
    if doc_row is None:
        # Chunk and document ids share an integer namespace; a common slip is
        # passing a chunk_id to --document-id. Hint toward --chunks only when the
        # unknown id is in fact a live chunk_id (silent otherwise — an id that's
        # genuinely unknown everywhere, or valid in both namespaces, carries no
        # detectable intent).
        extra = {}
        if _live_chunk_ids(conn, [args.document_id]):
            extra["hint"] = (
                f"chunk:{args.document_id} is a chunk_id — "
                f"did you mean --chunks chunk:{args.document_id}?"
            )
        raise SkillError(
            "DOCUMENT_NOT_FOUND",
            f"No document with id document:{args.document_id}.",
            **extra,
        )

    total = cur.execute(
        "SELECT COUNT(*) FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ?",
        (args.document_id,),
    ).fetchone()[0]

    rows = list(cur.execute(
        "SELECT chunk_id, chunk_index, section_heading, page_number, "
        "       content_type, text "
        "FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ? "
        "ORDER BY chunk_index LIMIT ? OFFSET ?",
        (args.document_id, args.limit, args.offset),
    ))

    # Every row here is a document-kind chunk of this one document, so the
    # source context is constant across the page. source_id is prefixed by kind;
    # document_id rides as a bare int and is tagged by the outer format pass.
    source = {
        "source_kind": "document",
        "source_id": format_source_id("document", doc_row[0]),
        "source_name": doc_row[1],
        "file_name": doc_row[1],
        "document_id": doc_row[0],
    }
    return format_output_ids({
        "mode": "document",
        "document": {"id": format_id("document", doc_row[0]), "file_name": doc_row[1]},
        "offset": args.offset,
        "limit": args.limit,
        "total": total,
        "preview": args.preview,
        "chunks": _chunks_from_rows(
            rows, args.preview, returning=args.returning, source=source,
        ),
    })


def _read_around_chunk(conn, args, *, session_id: int) -> dict:
    cur = conn.cursor()
    target = cur.execute(
        "SELECT chunk_id, source_kind, source_id, chunk_index "
        "FROM chunks WHERE chunk_id = ?",
        (args.around_chunk,),
    ).fetchone()
    if target is None:
        raise SkillError(
            "CHUNK_NOT_FOUND",
            f"No chunk with id chunk:{args.around_chunk}.",
        )
    target_id, sk, sid, target_idx = target

    # Memory wall: refuse a window centred on another session's finding chunk.
    # The window never crosses (source_kind, source_id), so gating the target
    # also keeps foreign finding chunks out of the neighbourhood.
    if sk == "finding":
        assert_findings_accessible(conn, session_id, [sid], action="read")

    rows = list(cur.execute(
        "SELECT chunk_id, chunk_index, section_heading, page_number, "
        "       content_type, text "
        "FROM chunks "
        "WHERE source_kind = ? AND source_id = ? "
        "  AND chunk_index BETWEEN ? AND ? "
        "ORDER BY chunk_index",
        (sk, sid, target_idx - args.window, target_idx + args.window),
    ))

    name = source_names(conn, {(sk, sid)}).get((sk, sid), "")
    # The window never crosses (source_kind, source_id), so the source context
    # is constant across the neighbourhood. file_name resolves off the target
    # (None for findings); document_id is honest-null off non-document chunks.
    file_name = chunk_locations(conn, [target_id]).get(
        target_id, {"file_name": None},
    )["file_name"]
    source = {
        "source_kind": sk,
        "source_id": format_source_id(sk, sid),
        "source_name": name,
        "file_name": file_name,
        "document_id": sid if sk == "document" else None,
    }
    return format_output_ids({
        "mode": "around",
        "target": {
            "chunk_id": target_id,
            "chunk_index": target_idx,
            "source_kind": sk,
            "source_id": format_source_id(sk, sid),
            "source_name": name,
        },
        "window": args.window,
        "preview": args.preview,
        "chunks": _chunks_from_rows(
            rows, args.preview, returning=args.returning, source=source,
        ),
    })


def work(*, conn, args, session_id) -> dict:
    # Reject a typo'd --returning field up front, so a not-found id set still
    # returns UNKNOWN_RETURNING_FIELD rather than a silent empty result.
    validate_returning(args.returning, CHUNK_FIELDS)
    mem = memory_enabled(conn, session_id)
    if args.chunk_ids is not None:
        return _read_by_chunk_ids(
            conn, args.chunk_ids, args.preview, mem=mem, session_id=session_id,
            returning=args.returning,
        )
    if args.around_chunk is not None:
        return _read_around_chunk(conn, args, session_id=session_id)
    return _read_by_document(conn, args)


def main(argv: list[str] | None = None) -> None:
    run(tool_name="read_chunks", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
