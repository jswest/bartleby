#!/usr/bin/env python3
"""search — keyword + semantic + RRF search across documents/summaries/findings.

See SPEC §6.2 for the full contract. Defaults: documents-only, both modes on,
context=1, limit=20.

Output:
    {
      "query": str,
      "modes": [str, ...],
      "source_kinds": [str, ...],
      "memory_excluded": bool,
      "context": int,
      "results": [{
        "chunk_id": int, "source_kind": str, "source_id": int,
        "source_name": str, "chunk_index": int,
        "section_heading": str|null, "content_type": str|null,
        "text": str,
        "context_before": [str, ...], "context_after": [str, ...],
        "score": float,
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
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--full-text", action="store_true", dest="full_text")
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
    return explicit or ["document"]


def _resolve_modes(args: argparse.Namespace) -> list[str]:
    if not args.semantic and not args.full_text:
        return ["semantic", "full-text"]
    modes = []
    if args.semantic:
        modes.append("semantic")
    if args.full_text:
        modes.append("full-text")
    return modes


def _fts_query(query: str) -> str:
    """Wrap each token in quotes so FTS5 treats it as a literal."""
    pieces = []
    for token in re.findall(r"\S+", query):
        clean = token.replace('"', "")
        if clean:
            pieces.append(f'"{clean}"')
    return " ".join(pieces)


def _fts_search(
    conn,
    query: str,
    source_kinds: list[str],
    limit: int,
) -> list[int]:
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    placeholders = ",".join("?" * len(source_kinds))
    rows = conn.cursor().execute(
        f"SELECT chunks_fts.rowid "
        f"FROM chunks_fts "
        f"JOIN chunks ON chunks.chunk_id = chunks_fts.rowid "
        f"WHERE chunks_fts MATCH ? AND chunks.source_kind IN ({placeholders}) "
        f"ORDER BY chunks_fts.rank "
        f"LIMIT ?",
        [fts_query, *source_kinds, limit],
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
    source_kinds: list[str],
    limit: int,
) -> list[int]:
    placeholders = ",".join("?" * len(source_kinds))
    rows = conn.cursor().execute(
        f"WITH nn AS ( "
        f"  SELECT rowid AS chunk_id, distance FROM chunks_vec "
        f"  WHERE embedding MATCH ? AND k = ? "
        f"  ORDER BY distance "
        f") "
        f"SELECT nn.chunk_id FROM nn "
        f"JOIN chunks c ON c.chunk_id = nn.chunk_id "
        f"WHERE c.source_kind IN ({placeholders}) "
        f"ORDER BY nn.distance "
        f"LIMIT ?",
        [query_bytes, limit, *source_kinds, limit],
    )
    return [row[0] for row in rows]


def _rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in rankings:
        for rank, chunk_id in enumerate(lst, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _single_mode_scored(ranking: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    return [
        (chunk_id, 1.0 / (k + rank))
        for rank, chunk_id in enumerate(ranking, start=1)
    ]


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


def _source_names(conn, source_keys: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    by_kind: dict[str, set[int]] = {}
    for kind, sid in source_keys:
        by_kind.setdefault(kind, set()).add(sid)
    cur = conn.cursor()
    for kind, ids in by_kind.items():
        ids_list = list(ids)
        ph = ",".join("?" * len(ids_list))
        if kind == "document":
            for did, fname in cur.execute(
                f"SELECT document_id, file_name FROM documents "
                f"WHERE document_id IN ({ph})",
                ids_list,
            ):
                out[("document", did)] = fname
        elif kind == "summary":
            for sid, fname in cur.execute(
                f"SELECT s.summary_id, d.file_name "
                f"FROM summaries s "
                f"JOIN documents d USING (document_id) "
                f"WHERE s.summary_id IN ({ph})",
                ids_list,
            ):
                out[("summary", sid)] = f"summary of {fname}"
        elif kind == "finding":
            for fid, title in cur.execute(
                f"SELECT finding_id, title FROM findings "
                f"WHERE finding_id IN ({ph})",
                ids_list,
            ):
                out[("finding", fid)] = title
    return out


def work(*, conn, args, session_id) -> dict:
    if not args.query or not args.query.strip():
        raise SkillError("EMPTY_QUERY", "Query must be non-empty.")

    source_kinds = _resolve_source_kinds(args)
    modes = _resolve_modes(args)

    memory_excluded = False
    memory_enabled = conn.cursor().execute(
        "SELECT memory_enabled FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    if "finding" in source_kinds and not memory_enabled:
        source_kinds = [k for k in source_kinds if k != "finding"]
        memory_excluded = True

    if not source_kinds:
        return {
            "query": args.query,
            "modes": modes,
            "source_kinds": [],
            "memory_excluded": memory_excluded,
            "context": args.context,
            "results": [],
        }

    overfetch = max(args.limit * OVERFETCH_MULTIPLIER, OVERFETCH_FLOOR)

    rankings: list[list[int]] = []
    if "full-text" in modes:
        rankings.append(_fts_search(conn, args.query, source_kinds, overfetch))
    if "semantic" in modes:
        query_bytes = _embed_query(args.query)
        rankings.append(_semantic_search(conn, query_bytes, source_kinds, overfetch))

    if len(rankings) > 1:
        scored = _rrf(rankings)
    elif len(rankings) == 1:
        scored = _single_mode_scored(rankings[0])
    else:
        scored = []
    scored = scored[: args.limit]

    if not scored:
        return {
            "query": args.query,
            "modes": modes,
            "source_kinds": source_kinds,
            "memory_excluded": memory_excluded,
            "context": args.context,
            "results": [],
        }

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

    source_keys = {(rows[cid][1], rows[cid][2]) for cid in chunk_ids if cid in rows}
    names = _source_names(conn, source_keys)

    results = []
    for chunk_id, score in scored:
        row = rows.get(chunk_id)
        if row is None:
            continue
        _, source_kind, source_id, chunk_index, section_heading, content_type, text = row
        before, after = _fetch_context(
            conn, source_kind, source_id, chunk_index, args.context
        )
        results.append({
            "chunk_id": chunk_id,
            "source_kind": source_kind,
            "source_id": source_id,
            "source_name": names.get((source_kind, source_id), ""),
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "content_type": content_type,
            "text": text,
            "context_before": before,
            "context_after": after,
            "score": score,
        })

    return {
        "query": args.query,
        "modes": modes,
        "source_kinds": source_kinds,
        "memory_excluded": memory_excluded,
        "context": args.context,
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    run(tool_name="search", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
