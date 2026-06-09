#!/usr/bin/env python
"""Benchmark local Ollama models against Bartleby's summarization prompt.

Samples documents from a real Bartleby project DB, runs each candidate model
N times per document, and writes one JSONL record per (model, document, run)
with timings and the parsed summary. Uses bartleby's prompt +
``DocumentSummary`` schema verbatim, at production settings (temp 0.0, the
default ``max_summarize_tokens``), so the benchmark exercises the same path
``bartleby ingest`` takes.

The whole (model, document, run) matrix is shuffled into one randomized order
so thermal throttling and weight-eviction can't systematically favor any
model or document.

Model set is auto-discovered from ``ollama list`` minus a skip-list of
non-text models (vision ``*-vl*``, embeddings ``nomic-embed*``, image-gen
``x/*``, coder variants), or pinned with ``--models``.

No warmup phase: with randomized ordering, models get evicted from memory
between calls anyway, so a per-model warmup is mostly wasted. Use
``wall_seconds - load_duration_ns/1e9`` (or ``eval_duration_ns``) for
load-corrected timing comparisons.

Example:
  uv run python benchmarks/summarize_local.py \\
      --db ~/.bartleby/projects/centralhudson-redux/bartleby.db \\
      --sample 12 \\
      --runs 3 \\
      --out benchmarks/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import ollama
from pydantic import ValidationError

from bartleby.providers.base import DocumentSummary
from bartleby.providers.prompt import build_summary_messages


# Production parity: bartleby summarizes at temperature 0.0 and truncates to
# DEFAULT_MAX_SUMMARIZE_TOKENS (bartleby/commands/config.py). Mirror both so
# the benchmark measures models under the settings ingest actually uses.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 50_000

# Substrings that mark a locally-installed model as not a text summarizer.
# Auto-discovery drops any model whose name contains one of these. Coder
# variants are skipped by default (they summarize poorly) but can be opted
# back in with --include-coder.
NON_TEXT_MARKERS = ("-vl", "vl:", "nomic-embed", "embed", "x/")
CODER_MARKERS = ("coder",)


def discover_models(client: ollama.Client, *, include_coder: bool) -> list[str]:
    """Names of installed Ollama models that are plausible text summarizers."""
    names = sorted(m.model for m in client.list().models)
    skip = NON_TEXT_MARKERS if include_coder else NON_TEXT_MARKERS + CODER_MARKERS
    return [n for n in names if not any(marker in n for marker in skip)]


def _document_lengths(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    """(document_id, token_count) for every document, shortest first."""
    rows = conn.execute(
        "SELECT document_id, COALESCE(token_count, 0) FROM documents ORDER BY token_count"
    ).fetchall()
    return [(int(d), int(t)) for d, t in rows]


def sample_documents(
    db_path: Path, *, sample: int, max_tokens: int, rng: random.Random
) -> list[int]:
    """Stratified sample of ``sample`` document ids, spread across length bands.

    Splits documents into short/medium/long tertiles by token_count and draws
    evenly from each. Deliberately forces in at least one document that exceeds
    ``max_tokens`` (when the corpus has one) so truncation behavior is always
    exercised.
    """
    conn = sqlite3.connect(db_path)
    try:
        docs = _document_lengths(conn)
    finally:
        conn.close()
    if not docs:
        raise SystemExit(f"No documents in {db_path}")
    if sample >= len(docs):
        return [d for d, _ in docs]

    third = len(docs) // 3 or 1
    bands = [docs[:third], docs[third : 2 * third], docs[2 * third :]]
    per_band = max(sample // 3, 1)

    chosen: list[int] = []
    for band in bands:
        ids = [d for d, _ in band]
        chosen.extend(rng.sample(ids, min(per_band, len(ids))))

    # Guarantee a truncation case if the corpus has one.
    over = [d for d, t in docs if t > max_tokens]
    if over and not any(d in over for d in chosen):
        chosen[-1] = rng.choice(over)

    # Top up / trim to exactly `sample`, drawing any remaining ids at random.
    remaining = [d for d, _ in docs if d not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < sample and remaining:
        chosen.append(remaining.pop())
    return chosen[:sample]


def load_source_text(db_path: Path, document_id: int) -> str:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT text FROM chunks "
            "WHERE source_kind='document' AND source_id=? "
            "ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise SystemExit(f"No chunks found for document {document_id} in {db_path}")
    return "\n\n".join(r[0] for r in rows)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text, len(ids)
    return tok.decode(ids[:max_tokens]), len(ids)


def _extract_timings(response) -> dict:
    g = lambda k: getattr(response, k, None)
    eval_count = g("eval_count")
    eval_duration_ns = g("eval_duration")
    tps = None
    if eval_count and eval_duration_ns:
        tps = eval_count / (eval_duration_ns / 1e9)
    return {
        "total_duration_ns": g("total_duration"),
        "load_duration_ns": g("load_duration"),
        "prompt_eval_count": g("prompt_eval_count"),
        "prompt_eval_duration_ns": g("prompt_eval_duration"),
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "tokens_per_second": tps,
    }


def call_summarize(
    client: ollama.Client,
    model: str,
    document_text: str,
    temperature: float,
) -> dict:
    """One Ollama summarize call. Returns timings + parsed summary, or error info."""
    wall_start = time.perf_counter()
    try:
        response = client.chat(
            model=model,
            messages=build_summary_messages(document_text),
            format=DocumentSummary.model_json_schema(),
            options={"temperature": temperature},
        )
    except Exception as e:
        return {
            "ok": False,
            "wall_seconds": time.perf_counter() - wall_start,
            "error": f"{type(e).__name__}: {e}",
        }
    wall_seconds = time.perf_counter() - wall_start

    raw = response.message.content or ""
    timings = _extract_timings(response)
    try:
        summary = DocumentSummary.model_validate_json(raw)
    except ValidationError as e:
        return {
            "ok": False,
            "wall_seconds": wall_seconds,
            "error": f"schema validation failed: {e}",
            "raw_output": raw,
            **timings,
        }
    return {
        "ok": True,
        "wall_seconds": wall_seconds,
        "summary": summary.model_dump(),
        **timings,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, required=True, help="Path to a bartleby SQLite DB")
    p.add_argument("--out", type=Path, required=True, help="JSONL output path")
    docs = p.add_mutually_exclusive_group()
    docs.add_argument(
        "--sample", type=int, default=12,
        help="Stratified sample of N documents across length bands (default 12)",
    )
    docs.add_argument(
        "--document-ids", type=int, nargs="+",
        help="Explicit document ids to benchmark (overrides --sample)",
    )
    p.add_argument("--runs", type=int, default=3, help="Measured runs per (model, doc) (default 3)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}, production parity)")
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Truncate source text to this many cl100k tokens (default {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Models to test (default: auto-discover from `ollama list` minus the skip-list)",
    )
    p.add_argument(
        "--include-coder", action="store_true",
        help="Keep coder models in auto-discovery (skipped by default)",
    )
    p.add_argument(
        "--ollama-host", default=None,
        help="Override OLLAMA_API_BASE (default http://localhost:11434)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the doc sample + run-order shuffle so the plan is reproducible",
    )
    args = p.parse_args(argv)

    rng = random.Random(args.seed)

    client = ollama.Client(host=args.ollama_host or "http://localhost:11434")
    models = args.models or discover_models(client, include_coder=args.include_coder)
    if not models:
        raise SystemExit("No candidate models found (auto-discovery returned nothing).")

    if args.document_ids:
        doc_ids = args.document_ids
    else:
        doc_ids = sample_documents(
            args.db, sample=args.sample, max_tokens=args.max_tokens, rng=rng
        )

    # Pre-load + truncate each document once; reuse the text across all runs.
    docs_text: dict[int, tuple[str, int, bool]] = {}
    for doc_id in doc_ids:
        source = load_source_text(args.db, doc_id)
        text, n_tokens = truncate_to_tokens(source, args.max_tokens)
        docs_text[doc_id] = (text, n_tokens, n_tokens > args.max_tokens)

    plan = [(m, d, i) for m in models for d in doc_ids for i in range(args.runs)]
    rng.shuffle(plan)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")

    def write(record: dict):
        record["timestamp"] = time.time()
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

    write({
        "kind": "config",
        "db": str(args.db),
        "document_ids": doc_ids,
        "documents": [
            {"document_id": d, "source_token_count": docs_text[d][1],
             "source_truncated": docs_text[d][2]}
            for d in doc_ids
        ],
        "models": models,
        "runs_per_model": args.runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "shuffled_plan": [{"model": m, "document_id": d, "run": i} for m, d, i in plan],
        "seed": args.seed,
    })

    print(f"Running {len(plan)} call(s): {len(models)} model(s) × "
          f"{len(doc_ids)} doc(s) × {args.runs} run(s)...", file=sys.stderr)
    for idx, (model, doc_id, run_idx) in enumerate(plan, 1):
        print(f"  [{idx}/{len(plan)}] {model} · doc {doc_id} (run {run_idx})",
              file=sys.stderr, flush=True)
        text, _, _ = docs_text[doc_id]
        result = call_summarize(client, model, text, args.temperature)
        write({"kind": "run", "model": model, "document_id": doc_id,
               "run_index": run_idx, **result})

    out_f.close()
    print(f"\nWrote results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
