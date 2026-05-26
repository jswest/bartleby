"""Helpers shared across skill scripts."""

from __future__ import annotations

import argparse
import re
from typing import Callable

from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_finding_chunks,
)
from bartleby.ingest.chunk import chunk_markdown_string
from bartleby.ingest.embed import embed_texts
from bartleby.skill_runner import SkillError


# Inline citation marker: standard markdown footnote syntax with a chunk_id.
_CITATION_MARKER = re.compile(r"\[\^(\d+)\]")


def extract_citations(body: str) -> list[int]:
    """Return chunk_ids from ``[^N]`` markers in first-appearance order, deduped."""
    seen: dict[int, None] = {}
    for m in _CITATION_MARKER.finditer(body):
        seen[int(m.group(1))] = None
    return list(seen)


def validate_chunk_ids_exist(conn, chunk_ids: list[int]) -> None:
    """Raise ``UNKNOWN_CITATIONS`` if any chunk_id is missing from ``chunks``."""
    if not chunk_ids:
        return
    ph = ",".join("?" * len(chunk_ids))
    cur = conn.cursor()
    seen = {
        row[0] for row in cur.execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({ph})", chunk_ids,
        )
    }
    missing = sorted(set(chunk_ids) - seen)
    if missing:
        raise SkillError(
            "UNKNOWN_CITATIONS",
            f"Inline citations reference unknown chunk_ids: {missing}. "
            "Each [^N] marker must be a real chunk_id in this project.",
            unknown_chunk_ids=missing,
        )


def rebuild_finding_chunks(conn, finding_id: int, body: str) -> list[int]:
    """Replace this finding's chunks with freshly chunked + embedded ones.

    Deletes any existing finding chunks, chunks the body, embeds each chunk,
    inserts them via the typed helper, and returns the new chunk_ids in
    insertion order. Callers also need to manage ``finding_citations``.
    """
    delete_chunks_for(conn, "finding", finding_id)
    rows = chunk_markdown_string(body)
    if not rows:
        return []
    embeddings = embed_texts([r.text for r in rows])
    chunk_inputs = [
        ChunkInput(
            text=row.text,
            embedding=emb,
            chunk_index=i,
            section_heading=row.section_heading,
            content_type=row.content_type,
        )
        for i, (row, emb) in enumerate(zip(rows, embeddings))
    ]
    return insert_finding_chunks(conn, finding_id, chunk_inputs)


def replace_finding_citations(conn, finding_id: int, chunk_ids: list[int]) -> None:
    """Atomically swap ``finding_citations`` rows for this finding."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM finding_citations WHERE finding_id = ?", (finding_id,),
    )
    cur.executemany(
        "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
        [(finding_id, cid) for cid in chunk_ids],
    )


def resolve_citations(conn, chunk_ids: list[int]) -> list[dict]:
    """Enrich each cited chunk_id with source_name/file_name/page_number.

    The agent gets this back so it can render human-readable citations in its
    reply alongside the structural chunk_id.
    """
    if not chunk_ids:
        return []
    locations = chunk_locations(conn, chunk_ids)
    names = source_names(
        conn, {(loc["source_kind"], loc["source_id"]) for loc in locations.values()},
    )
    out = []
    for cid in chunk_ids:
        loc = locations.get(cid)
        if loc is None:    # citation chunk vanished between validation and here
            continue
        out.append({
            "chunk_id": cid,
            "source_kind": loc["source_kind"],
            "source_name": names.get((loc["source_kind"], loc["source_id"]), ""),
            "file_name": loc["file_name"],
            "page_number": loc["page_number"],
        })
    return out


def comma_int_list(label: str) -> Callable[[str], list[int]]:
    """argparse ``type=`` factory for a comma-separated list of ints.

    ``label`` names the id being parsed (e.g. ``"document_id"``) so the error
    message is meaningful to the agent reading argparse's failure.
    """
    def _parse(s: str) -> list[int]:
        out: list[int] = []
        for piece in s.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                out.append(int(piece))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"'{piece}' is not an integer {label}"
                ) from None
        if not out:
            raise argparse.ArgumentTypeError(f"at least one {label} required")
        return out
    return _parse


def source_names(
    conn, source_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], str]:
    """Resolve display names for ``(source_kind, source_id)`` pairs in one batch.

    Documents → file_name; summaries → ``"summary of <file_name>"``; findings →
    finding title; images → ``"image in <file_name>, p.<N>"`` (with
    ``" (+K other docs)"`` if the image appears in multiple documents).
    Pairs that don't resolve (deleted underneath us) are simply absent from
    the returned dict.
    """
    by_kind: dict[str, list[int]] = {}
    for kind, sid in source_keys:
        by_kind.setdefault(kind, []).append(sid)

    out: dict[tuple[str, int], str] = {}
    cur = conn.cursor()
    for kind, ids in by_kind.items():
        ph = ",".join("?" * len(ids))
        if kind == "document":
            for did, fname in cur.execute(
                f"SELECT document_id, file_name FROM documents "
                f"WHERE document_id IN ({ph})",
                ids,
            ):
                out[("document", did)] = fname
        elif kind == "summary":
            for sid, fname in cur.execute(
                f"SELECT s.summary_id, d.file_name "
                f"FROM summaries s JOIN documents d USING (document_id) "
                f"WHERE s.summary_id IN ({ph})",
                ids,
            ):
                out[("summary", sid)] = f"summary of {fname}"
        elif kind == "finding":
            for fid, title in cur.execute(
                f"SELECT finding_id, title FROM findings "
                f"WHERE finding_id IN ({ph})",
                ids,
            ):
                out[("finding", fid)] = title
        elif kind == "image":
            out.update(_image_source_names(cur, ids, ph))
    return out


def chunk_locations(
    conn, chunk_ids: list[int]
) -> dict[int, dict]:
    """Resolve {source_kind, source_id, file_name, page_number} per chunk_id.

    ``file_name`` / ``page_number`` may be ``None`` when not applicable:
      - document chunks: file_name from the document; page_number parsed from
        ``section_heading`` (the 'page N' convention pdfplumber writes).
      - summary chunks: file_name from the underlying document; page_number is
        always None (summaries aren't paginated).
      - image chunks: file_name from the primary document the image is linked
        to; page_number from that ``document_images`` join row.
      - finding chunks: file_name + page_number both None.

    Chunks whose row was deleted between query and resolution are absent from
    the result entirely.
    """
    if not chunk_ids:
        return {}
    ph = ",".join("?" * len(chunk_ids))
    cur = conn.cursor()
    rows = list(cur.execute(
        f"SELECT chunk_id, source_kind, source_id, page_number "
        f"FROM chunks WHERE chunk_id IN ({ph})",
        chunk_ids,
    ))

    out: dict[int, dict] = {}
    by_kind: dict[str, list[int]] = {}
    for cid, kind, sid, page_number in rows:
        out[cid] = {
            "source_kind": kind, "source_id": sid,
            "file_name": None, "page_number": page_number,
        }
        by_kind.setdefault(kind, []).append(cid)

    # Resolve file_name per source_kind. page_number is already on the row
    # except for image chunks, where the source-of-truth is the
    # document_images join (an image can live on different pages in different
    # documents).
    for kind, cids in by_kind.items():
        sids = list({out[cid]["source_id"] for cid in cids})
        sid_ph = ",".join("?" * len(sids))
        if kind == "document":
            names = {
                did: fname for did, fname in cur.execute(
                    f"SELECT document_id, file_name FROM documents "
                    f"WHERE document_id IN ({sid_ph})",
                    sids,
                )
            }
            for cid in cids:
                out[cid]["file_name"] = names.get(out[cid]["source_id"])
        elif kind == "summary":
            names = {
                sid: fname for sid, fname in cur.execute(
                    f"SELECT s.summary_id, d.file_name "
                    f"FROM summaries s JOIN documents d USING (document_id) "
                    f"WHERE s.summary_id IN ({sid_ph})",
                    sids,
                )
            }
            for cid in cids:
                out[cid]["file_name"] = names.get(out[cid]["source_id"])
        elif kind == "image":
            anchors = _image_anchors(cur, sids)
            for cid in cids:
                anchor = anchors.get(out[cid]["source_id"])
                if anchor:
                    out[cid]["file_name"] = anchor["file_name"]
                    out[cid]["page_number"] = anchor["page_number"]
        # 'finding' falls through with file_name=None (and page_number from
        # the column, which is None for finding chunks).
    return out


def _image_anchors(cur, image_ids: list[int]) -> dict[int, dict]:
    """Per-image primary anchor + count of additional documents using it.

    Returns ``{image_id: {file_name, page_number, other_doc_count}}``. The
    'primary' anchor is the lowest ``(document_id, page_number)`` join row,
    matching the existing source_name formatting rule.
    """
    if not image_ids:
        return {}
    ph = ",".join("?" * len(image_ids))
    rows = list(cur.execute(
        f"SELECT di.image_id, di.document_id, di.page_number, d.file_name "
        f"FROM document_images di "
        f"JOIN documents d ON d.document_id = di.document_id "
        f"WHERE di.image_id IN ({ph}) "
        f"ORDER BY di.image_id, di.document_id, di.page_number",
        image_ids,
    ))
    by_image: dict[int, list[tuple[int, int | None, str]]] = {}
    for image_id, doc_id, page_number, file_name in rows:
        by_image.setdefault(image_id, []).append((doc_id, page_number, file_name))

    out: dict[int, dict] = {}
    for image_id, occurrences in by_image.items():
        primary_doc, primary_page, primary_name = occurrences[0]
        out[image_id] = {
            "file_name": primary_name,
            "page_number": primary_page,
            "other_doc_count": len({d for d, _, _ in occurrences}) - 1,
        }
    return out


def _image_source_names(cur, ids, ph) -> dict[tuple[str, int], str]:
    """Format per-image display names from `_image_anchors`."""
    out: dict[tuple[str, int], str] = {}
    for image_id, anchor in _image_anchors(cur, ids).items():
        page_str = (f", p.{anchor['page_number']}"
                    if anchor["page_number"] is not None else "")
        suffix = (f" (+{anchor['other_doc_count']} other docs)"
                  if anchor["other_doc_count"] > 0 else "")
        out[("image", image_id)] = f"image in {anchor['file_name']}{page_str}{suffix}"
    return out
