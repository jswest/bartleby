#!/usr/bin/env python3
"""search — keyword + semantic + RRF search across documents/summaries/findings/images.

Defaults: documents + images, both modes on, no surrounding context, limit=20.
Pass any of ``--documents`` / ``--summaries`` / ``--findings`` / ``--images``
to override the default set. Opt into neighbor chunks with ``--add-context N``
(0..5) — each step roughly multiplies output size by (1 + 2N).

``--tag <name>`` (repeatable, OR semantics) restricts to chunks whose
underlying document carries any of the given tags. ``--file-like <pattern>``
(SQL ``LIKE``, repeatable for OR) restricts to chunks whose document's
``file_name`` matches a pattern. Both combine naturally with ``--in-documents``
(intersection: the document must be in every active set) and drop findings the
same way (findings have no document anchor).

Whenever a scope filter (``--in-documents`` / ``--tag`` / ``--file-like``) is
active the response carries a nested ``filters`` object echoing it — ``{tags,
in_documents, file_like, authored_after, authored_before, include_nulls,
excluded_null_dated}`` — the same contract ``scan`` / ``list_documents`` /
``describe_corpus`` emit (search takes no date bounds, so those keys are
null/0). It is absent on an unfiltered search; the query terms themselves always
stay top-level under ``query``.

Output:
    {
      "query": str,
      "modes": [str, ...],
      "source_kinds": [str, ...],
      "memory_excluded": bool,
      "context": int,
      "results": [{
        "chunk_id": int, "source_kind": str, "source_id": int,
        "source_name": str,
        "file_name": str|null,        # the originating doc (None for findings)
        "page_number": int|null,      # first-class column; populated for pdfplumber + image chunks
        "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
        # context_before/context_after are present only with --add-context > 0;
        # omitted entirely at the default --add-context 0.
        "context_before": [{"chunk_id": int, "chunk_index": int, "text": str}, ...],
        "context_after":  [{"chunk_id": int, "chunk_index": int, "text": str}, ...],
        "rank": int,                  # 1-indexed within this query's results
        "score": float,               # raw RRF score (small, don't compare across queries)
        "normalized_score": float,    # top hit = 1.0, others scaled to that
        # image hits only:
        "image_id": int, "image_file_path": str,
      }, ...]
    }

With ``--brief`` each hit is trimmed to a triage projection — ``chunk_id``,
``source_kind``, ``source_name``, ``page_number``, ``rank``,
``normalized_score``, and a truncated ``text`` preview — dropping ``source_id``,
``chunk_index``, ``section_heading``, ``content_type``, ``score``, the full
``text``, the context arrays, and the image locators. The envelope is unchanged;
``--add-context`` is ignored under ``--brief``.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess

from bartleby.db.schema import EMBEDDING_DIM
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import (
    add_file_like_arg, apply_preview, chunk_locations, comma_int_list,
    memory_enabled, positive_int, source_names,
)
from bartleby.skill_scripts._tags import resolve_scope


RRF_K = 60
DEFAULT_CONTEXT = 0
DEFAULT_LIMIT = 20
MAX_CONTEXT = 5
OVERFETCH_MULTIPLIER = 5
OVERFETCH_FLOOR = 50
BRIEF_PREVIEW_CHARS = 240


def _context_value(s: str) -> int:
    v = int(s)
    if not 0 <= v <= MAX_CONTEXT:
        raise argparse.ArgumentTypeError(
            f"add-context must be in 0..{MAX_CONTEXT}"
        )
    return v


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("search", __doc__)
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
    p.add_argument(
        "--tag", action="append", default=None, dest="tags",
        help=(
            "Restrict to documents carrying this tag. Repeat for OR "
            "semantics. Findings are dropped. Combines with --in-documents "
            "as an intersection."
        ),
    )
    add_file_like_arg(p)
    p.add_argument(
        "--add-context",
        type=_context_value,
        default=DEFAULT_CONTEXT,
        dest="context",
        help=(
            "Number of neighbor chunks (0..5) to attach to each hit as "
            "context_before/context_after. Default 0 (keys omitted entirely). "
            "Each step roughly multiplies output size by (1 + 2N) — use sparingly."
        ),
    )
    p.add_argument("--limit", type=positive_int, default=DEFAULT_LIMIT)
    p.add_argument(
        "--brief",
        action="store_true",
        help=(
            "Skinny per-hit projection for triage: chunk_id, source_kind, "
            "source_name, page_number, rank, normalized_score, and a truncated "
            f"text preview ({BRIEF_PREVIEW_CHARS} chars). Drops source_id, "
            "chunk_index, section_heading, content_type, full text, the context "
            "arrays, and image locators. --add-context is ignored under --brief."
        ),
    )
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
    # Push the scope predicate into the KNN via ``rowid IN (subquery)`` so the
    # k neighbors vec0 returns are the k nearest *in-scope* chunks. A bare
    # post-filter (fetch k globally-nearest, then drop out-of-scope rows)
    # collapses to ~nothing whenever the scope is a small slice of the corpus,
    # because the globally-nearest chunks rarely fall inside it — issue #55.
    # vec0 accepts ``rowid IN`` alongside MATCH/k as of sqlite-vec 0.1.x.
    # ``k`` already caps the row count, so no outer LIMIT is needed.
    scope_sql, scope_params = _scope_clause(scope)
    rows = conn.cursor().execute(
        f"SELECT rowid FROM chunks_vec "
        f"WHERE embedding MATCH ? AND k = ? "
        f"  AND rowid IN (SELECT chunk_id FROM chunks WHERE {scope_sql}) "
        f"ORDER BY distance",
        [query_bytes, limit, *scope_params],
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
) -> tuple[list[dict], list[dict]]:
    """Return neighbor chunks as {chunk_id, chunk_index, text} dicts so the
    agent can fetch a neighbor directly via ``read_chunks --chunks <id>``
    instead of guessing the id from chunk_index ordering.

    Only called when ``context > 0`` — the caller omits the context keys
    entirely at ``--add-context 0``."""
    lo = hit_index - context
    hi = hit_index + context
    rows = list(conn.cursor().execute(
        "SELECT chunk_id, chunk_index, text FROM chunks "
        "WHERE source_kind = ? AND source_id = ? AND chunk_index BETWEEN ? AND ? "
        "ORDER BY chunk_index",
        (source_kind, source_id, lo, hi),
    ))
    before = [
        {"chunk_id": cid, "chunk_index": idx, "text": text}
        for cid, idx, text in rows if idx < hit_index
    ]
    after = [
        {"chunk_id": cid, "chunk_index": idx, "text": text}
        for cid, idx, text in rows if idx > hit_index
    ]
    return before, after


def work(*, conn, args, session_id) -> dict:
    if not args.query or not args.query.strip():
        raise SkillError("EMPTY_QUERY", "Query must be non-empty.")

    source_kinds = _resolve_source_kinds(args)
    modes = _resolve_modes(args)
    scope = resolve_scope(
        conn, in_documents=args.in_documents, tags=args.tags,
        file_like=args.file_like,
    )
    # None = whole corpus, [] = a filter matched nothing, else the resolved slice.
    restrict = scope.document_ids

    # Findings drop out when memory is off OR when --in-documents/--tag is set
    # (findings have no document anchor). memory_excluded reports only the
    # memory case — that's the signal documented in SKILL.md.
    memory_excluded = (
        "finding" in source_kinds and not memory_enabled(conn, session_id)
    )
    drop_findings = memory_excluded or restrict is not None
    if drop_findings:
        source_kinds = [k for k in source_kinds if k != "finding"]

    def _response(results: list) -> dict:
        return scope.echo_into({
            "query": args.query,
            "modes": modes,
            "source_kinds": source_kinds,
            "memory_excluded": memory_excluded,
            "context": args.context,
            "results": results,
        })

    # No source kinds left, or a scope filter that matched nothing: zero hits.
    if not source_kinds or (restrict is not None and not restrict):
        return _response([])

    scope_dict = _build_scope(conn, source_kinds, restrict)
    overfetch = max(args.limit * OVERFETCH_MULTIPLIER, OVERFETCH_FLOOR)

    rankings: list[list[int]] = []
    if "full-text" in modes:
        rankings.append(_fts_search(conn, args.query, scope_dict, overfetch))
    if "semantic" in modes:
        rankings.append(
            _semantic_search(conn, _embed_query(args.query), scope_dict, overfetch)
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
        loc = locations.get(chunk_id, {"file_name": None, "page_number": None})
        if args.brief:
            results.append({
                "chunk_id": chunk_id,
                "source_kind": source_kind,
                "source_name": names[(source_kind, source_id)],
                "page_number": loc["page_number"],
                "rank": rank,
                "normalized_score": score / top_score,
                "text": apply_preview(text, BRIEF_PREVIEW_CHARS),
            })
            continue
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
            "rank": rank,
            "score": score,
            "normalized_score": score / top_score,
        }
        # Neighbor context is opt-in; at --add-context 0 (the triage default) the
        # keys are omitted entirely rather than shipped as empty arrays.
        if args.context > 0:
            before, after = _fetch_context(
                conn, source_kind, source_id, chunk_index, args.context
            )
            hit["context_before"] = before
            hit["context_after"] = after
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
