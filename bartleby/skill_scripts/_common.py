"""Helpers shared across skill scripts."""

from __future__ import annotations

import argparse
from typing import Callable


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


def _image_source_names(cur, ids, ph) -> dict[tuple[str, int], str]:
    """Build per-image display names from `document_images`.

    Picks the lowest (document_id, page_number) per image as the primary
    occurrence and appends ``"(+K other docs)"`` if the image is shared
    across multiple documents.
    """
    rows = list(cur.execute(
        f"SELECT di.image_id, di.document_id, di.page_number, d.file_name "
        f"FROM document_images di "
        f"JOIN documents d ON d.document_id = di.document_id "
        f"WHERE di.image_id IN ({ph}) "
        f"ORDER BY di.image_id, di.document_id, di.page_number",
        ids,
    ))
    by_image: dict[int, list[tuple[int, int | None, str]]] = {}
    for image_id, doc_id, page_number, file_name in rows:
        by_image.setdefault(image_id, []).append((doc_id, page_number, file_name))

    out: dict[tuple[str, int], str] = {}
    for image_id, occurrences in by_image.items():
        primary_doc, primary_page, primary_name = occurrences[0]
        page_str = f", p.{primary_page}" if primary_page is not None else ""
        n_other_docs = len({d for d, _, _ in occurrences}) - 1
        suffix = f" (+{n_other_docs} other docs)" if n_other_docs > 0 else ""
        out[("image", image_id)] = f"image in {primary_name}{page_str}{suffix}"
    return out
