#!/usr/bin/env python3
"""search — keyword + semantic + RRF search across documents/summaries/findings/images.

Defaults: documents + images, both modes on, context=1, limit=20. Pass any of
``--documents`` / ``--summaries`` / ``--findings`` / ``--images`` to override
the default set.

Output:
    {
      "query": str,
      "modes": [str, ...],
      "source_kinds": [str, ...],
      "memory_excluded": bool,
      "in_documents": [int, ...]|null,
      "context": int,
      "results": [{
        "chunk_id": int, "source_kind": str, "source_id": int,
        "source_name": str,
        "file_name": str|null,        # the originating doc (None for findings)
        "page_number": int|null,      # first-class column; populated for pdfplumber + image chunks
        "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
        "context_before": [str, ...], "context_after": [str, ...],
        "rank": int,                  # 1-indexed within this query's results
        "score": float,               # raw RRF score (small, don't compare across queries)
        "normalized_score": float,    # top hit = 1.0, others scaled to that
        # image hits only:
        "image_id": int, "image_file_path": str,
      }, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess

from bartleby.db.schema import EMBEDDING_DIM
from bartleby.skill_runner import SkillError, run
from bartleby.skill_scripts._common import (
    chunk_locations, comma_int_list, source_names,
)


RRF_K = 60
DEFAULT_CONTEXT = 1
DEFAULT_LIMIT = 20
MAX_CONTEXT = 5
OVERFETCH_MULTIPLIER = 5
OVERFETCH_FLOOR = 50


def _context_value(s: str) -> int:
    v = int(s)
    if not 0 <= v <= MAX_CONTEXT:
        raise argparse.ArgumentTypeError(
            f"context must be in 0..{MAX_CONTEXT}"
        )
    return v


def _positive_int(s: str) -> int:
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="search")
    p.add_argument("query", type=str)
    p.add_argument("--documents", action="store_true")
    p.add_argument("--summaries", action="store_true")
    p.add_argument("--findings", action="store_true")
    p.add_argument("--images", action="store_true")
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--full-text", action="store_true", dest="full_text")
    p.add_argument(
        "--in-documents",
        type=comma_int_list("document_id"),
        default=None,
        dest="in_documents",
        help=(
            "Comma-separated document_ids. Scopes the search to those "
            "documents' chunks and their summaries' chunks. Findings are "
            "dropped (they are not tied to documents)."
        ),
    )
    p.add_argument("--context", type=_context_value, default=DEFAULT_CONTEXT)
    p.add_argument("--limit", type=_positive_int, default=DEFAULT_LIMIT)
    p.add_argument("--project", type=str, default=None)
    return p.parse_args(argv)


def _resolve_source_kinds(args: argparse.Namespace) -> list[str]:
    explicit = []
    if args.documents:
        explicit.append("document")
    if args.summaries:
        explicit.append("summary")
    if args.findings:
        explicit.append("finding")
    if args.images:
        explicit.append("image")
    return explicit or ["document", "image"]


def _resolve_modes(args: argparse.Namespace) -> list[str]:
    if not args.semantic and not args.full_text:
        return ["semantic", "full-text"]
    return [m for flag, m in
            [(args.semantic, "semantic"), (args.full_text, "full-text")] if flag]


def _fts_query(query: str) -> str:
    """Wrap each token in quotes so FTS5 treats it as a literal."""
    pieces = []
    for token in re.findall(r"\S+", query):
        clean = token.replace('"', "")
        if clean:
            pieces.append(f'"{clean}"')
    return " ".join(pieces)


def _scope_clause(
    scope: dict[str, list[int] | None],
) -> tuple[str, list]:
    """Build a SQL predicate over ``chunks`` that matches any allowed
    (source_kind, source_id) combination.

    ``scope`` maps source_kind → either a list of allowed source_ids
    (filtered scope) or ``None`` (unrestricted — match all of that kind).
    """
    if not scope:
        return "0", []
    parts: list[str] = []
    params: list = []
    for kind, ids in scope.items():
        if ids is None:
            parts.append("chunks.source_kind = ?")
            params.append(kind)
        else:
            if not ids:
                continue
            placeholders = ",".join("?" * len(ids))
            parts.append(
                f"(chunks.source_kind = ? AND chunks.source_id IN ({placeholders}))"
            )
            params.append(kind)
            params.extend(ids)
    if not parts:
        return "0", []
    return "(" + " OR ".join(parts) + ")", params


def _build_scope(
    conn,
    source_kinds: list[str],
    in_documents: list[int] | None,
) -> dict[str, list[int] | None]:
    """Resolve the scope dict consumed by ``_scope_clause``.

    Without ``in_documents``, every requested kind is unrestricted.
    With ``in_documents``: 'document' is restricted to those ids; 'summary'
    is restricted to summaries whose document_id is in those ids; 'image' is
    restricted to images linked to those documents via ``document_images``;
    'finding' is excluded entirely (findings aren't tied to documents).
    """
    if in_documents is None:
        return {kind: None for kind in source_kinds}

    placeholders = ",".join("?" * len(in_documents))
    cur = conn.cursor()
    scope: dict[str, list[int] | None] = {}
    if "document" in source_kinds:
        scope["document"] = list(in_documents)
    if "summary" in source_kinds:
        scope["summary"] = [
            row[0] for row in cur.execute(
                f"SELECT summary_id FROM summaries "
                f"WHERE document_id IN ({placeholders})",
                in_documents,
            )
        ]
    if "image" in source_kinds:
        scope["image"] = [
            row[0] for row in cur.execute(
                f"SELECT DISTINCT image_id FROM document_images "
                f"WHERE document_id IN ({placeholders})",
                in_documents,
            )
        ]
    return scope


def _fts_search(
    conn,
    query: str,
    scope: dict[str, list[int] | None],
    limit: int,
) -> list[int]:
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    scope_sql, scope_params = _scope_clause(scope)
    rows = conn.cursor().execute(
        f"SELECT chunks_fts.rowid "
        f"FROM chunks_fts "
        f"JOIN chunks ON chunks.chunk_id = chunks_fts.rowid "
        f"WHERE chunks_fts MATCH ? AND {scope_sql} "
        f"ORDER BY chunks_fts.rank "
        f"LIMIT ?",
        [fts_query, *scope_params, limit],
    )
    return [row[0] for row in rows]


def _embed_query(query: str) -> bytes:
    """Shell out to ``bartleby embed`` via list-form subprocess (SPEC §5.5)."""
    result = subprocess.run(
        ["bartleby", "embed", query],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SkillError(
            "EMBED_FAILED",
            f"`bartleby embed` exited {result.returncode}: "
            f"{result.stderr.strip() or '(no stderr)'}",
        )
    try:
        vec = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise SkillError(
            "EMBED_FAILED", f"Could not parse `bartleby embed` output: {e}"
        ) from None
    if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
        raise SkillError(
            "EMBED_FAILED",
            f"`bartleby embed` returned {len(vec) if isinstance(vec, list) else type(vec).__name__}; "
            f"expected list of {EMBEDDING_DIM} floats.",
        )
    return struct.pack(f"{EMBEDDING_DIM}f", *vec)


def _semantic_search(
    conn,
    query_bytes: bytes,
    scope: dict[str, list[int] | None],
    limit: int,
) -> list[int]:
    # sqlite-vec's MATCH operator cannot accept extra WHERE predicates on the
    # vec0 virtual table, so we over-fetch nearest neighbors and filter them
    # in the outer query against the chunks table.
    scope_sql, scope_params = _scope_clause(scope)
    rows = conn.cursor().execute(
        f"WITH nn AS ( "
        f"  SELECT rowid AS chunk_id, distance FROM chunks_vec "
        f"  WHERE embedding MATCH ? AND k = ? "
        f"  ORDER BY distance "
        f") "
        f"SELECT nn.chunk_id FROM nn "
        f"JOIN chunks ON chunks.chunk_id = nn.chunk_id "
        f"WHERE {scope_sql} "
        f"ORDER BY nn.distance "
        f"LIMIT ?",
        [query_bytes, limit, *scope_params, limit],
    )
    return [row[0] for row in rows]


def _rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in rankings:
        for rank, chunk_id in enumerate(lst, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fetch_context(
    conn,
    source_kind: str,
    source_id: int,
    hit_index: int,
    context: int,
) -> tuple[list[str], list[str]]:
    if context <= 0:
        return [], []
    lo = hit_index - context
    hi = hit_index + context
    rows = list(conn.cursor().execute(
        "SELECT chunk_index, text FROM chunks "
        "WHERE source_kind = ? AND source_id = ? AND chunk_index BETWEEN ? AND ? "
        "ORDER BY chunk_index",
        (source_kind, source_id, lo, hi),
    ))
    before = [text for idx, text in rows if idx < hit_index]
    after = [text for idx, text in rows if idx > hit_index]
    return before, after


def work(*, conn, args, session_id) -> dict:
    if not args.query or not args.query.strip():
        raise SkillError("EMPTY_QUERY", "Query must be non-empty.")

    source_kinds = _resolve_source_kinds(args)
    modes = _resolve_modes(args)
    in_documents = args.in_documents

    memory_enabled = conn.cursor().execute(
        "SELECT memory_enabled FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    # Findings drop out when memory is off OR when --in-documents is set
    # (findings have no document anchor). memory_excluded reports only the
    # memory case — that's the signal documented in SKILL.md.
    memory_excluded = "finding" in source_kinds and not memory_enabled
    drop_findings = memory_excluded or in_documents is not None
    if drop_findings:
        source_kinds = [k for k in source_kinds if k != "finding"]

    def _response(results: list) -> dict:
        return {
            "query": args.query,
            "modes": modes,
            "source_kinds": source_kinds,
            "memory_excluded": memory_excluded,
            "in_documents": in_documents,
            "context": args.context,
            "results": results,
        }

    if not source_kinds:
        return _response([])

    scope = _build_scope(conn, source_kinds, in_documents)
    overfetch = max(args.limit * OVERFETCH_MULTIPLIER, OVERFETCH_FLOOR)

    rankings: list[list[int]] = []
    if "full-text" in modes:
        rankings.append(_fts_search(conn, args.query, scope, overfetch))
    if "semantic" in modes:
        rankings.append(
            _semantic_search(conn, _embed_query(args.query), scope, overfetch)
        )

    scored = _rrf(rankings)[: args.limit]
    if not scored:
        return _response([])

    chunk_ids = [cid for cid, _ in scored]
    placeholders = ",".join("?" * len(chunk_ids))
    rows = {
        row[0]: row
        for row in conn.cursor().execute(
            f"SELECT chunk_id, source_kind, source_id, chunk_index, "
            f"       section_heading, content_type, text "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
    }
    names = source_names(conn, {(r[1], r[2]) for r in rows.values()})
    image_paths = _image_paths(
        conn, [r[2] for r in rows.values() if r[1] == "image"]
    )
    locations = chunk_locations(conn, list(rows.keys()))

    top_score = scored[0][1]
    results = []
    for rank, (chunk_id, score) in enumerate(scored, start=1):
        _, source_kind, source_id, chunk_index, section_heading, content_type, text = rows[chunk_id]
        before, after = _fetch_context(
            conn, source_kind, source_id, chunk_index, args.context
        )
        loc = locations.get(chunk_id, {"file_name": None, "page_number": None})
        hit = {
            "chunk_id": chunk_id,
            "source_kind": source_kind,
            "source_id": source_id,
            "source_name": names[(source_kind, source_id)],
            "file_name": loc["file_name"],
            "page_number": loc["page_number"],
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "content_type": content_type,
            "text": text,
            "context_before": before,
            "context_after": after,
            "rank": rank,
            "score": score,
            "normalized_score": score / top_score,
        }
        if source_kind == "image":
            hit["image_id"] = source_id
            hit["image_file_path"] = image_paths.get(source_id, "")
        results.append(hit)

    return _response(results)


def _image_paths(conn, image_ids: list[int]) -> dict[int, str]:
    if not image_ids:
        return {}
    placeholders = ",".join("?" * len(image_ids))
    rows = conn.cursor().execute(
        f"SELECT image_id, file_path FROM images WHERE image_id IN ({placeholders})",
        image_ids,
    )
    return {row[0]: row[1] for row in rows}


def main(argv: list[str] | None = None) -> None:
    run(tool_name="search", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
