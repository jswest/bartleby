#!/usr/bin/env python
"""Benchmark local Ollama models against Bartleby's summarization prompt.

Benchmarks one committed PDF (``benchmarks/corpus/``), assembling its
summarization input **the same way production does**: extract text with the
pdfplumber backend, chunk it with Bartleby's chunker, and join the document
chunks in page/chunk order — exactly what ``writer.summary_input`` produces for
an image-free document. Then each candidate model summarizes that input N
times, at production settings (temp 0.0, the default ``max_summarize_tokens``),
using Bartleby's actual ``build_summary_messages`` + ``DocumentSummary`` schema.

The benchmark doc is **image-free on purpose**: with no image chunks, the
pdfplumber → chunk path reproduces the *whole* production summary input, so the
benchmark is faithful without dragging in the caption pipeline. (Image-bearing
docs, Docling extraction, and a multi-doc corpus are tracked as follow-ups.)

The (model, run) matrix is shuffled into one randomized order so thermal
throttling and weight-eviction can't systematically favor any model.

Model set is auto-discovered from ``ollama list`` minus a skip-list of
non-text models (vision ``*-vl*``, embeddings ``nomic-embed*``, image-gen
``x/*``, coder variants), or pinned with ``--models``.

Example:
  uv run python benchmarks/summarize_local.py --runs 3 --out benchmarks/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import ollama
from pydantic import ValidationError

from bartleby.ingest import pdfplumber as pdfplumber_pipeline
from bartleby.ingest.text import chunk_text
from bartleby.lib.consts import DEFAULT_OCR_MIN_CONFIDENCE, DEFAULT_SPARSE_TEXT_THRESHOLD
from bartleby.providers.base import DocumentSummary
from bartleby.providers.prompt import build_summary_messages


# Production parity: bartleby summarizes at temperature 0.0 and truncates to
# DEFAULT_MAX_SUMMARIZE_TOKENS (bartleby/commands/config.py). Mirror both so
# the benchmark measures models under the settings ingest actually uses.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 50_000

# The committed benchmark document. Image-free, medium-length PSC order; see
# benchmarks/README.md for why this one.
DEFAULT_PDF = Path(__file__).parent / "corpus" / \
    "0109_Order_Denying_Request_for_Rehearing_and_Reconsideration.pdf"

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


def build_summary_input(pdf_path: Path) -> tuple[str, dict]:
    """Reproduce production's summary input for an image-free PDF.

    Mirrors ``scribe._parse_pdf_pdfplumber`` + ``writer.summary_input``: extract
    with the pdfplumber backend at the production thresholds, chunk each text
    page with ``chunk_text``, and join the document chunks in page/chunk order
    with blank lines. Returns ``(text, meta)``; ``meta`` flags anything that
    would make this *not* a faithful image-free reproduction (image-routed or
    sparse pages), which the caller surfaces.
    """
    result = pdfplumber_pipeline.convert(
        pdf_path,
        sparse_text_threshold=DEFAULT_SPARSE_TEXT_THRESHOLD,
        ocr_min_confidence=DEFAULT_OCR_MIN_CONFIDENCE,
    )
    parts: list[str] = []
    image_pages: list[int] = []
    # if/elif/if is deliberate: a text page is chunked, else a sparse page would
    # route to the VLM — but *either* kind can also carry embedded images, so the
    # embedded-image check is a separate `if`, not part of the elif chain.
    for page in result.pages:
        if page.content_type is not None:
            parts.extend(chunk_text(page.text))
        elif page.page_render_png is not None:
            image_pages.append(page.page_number)  # would route to the VLM in prod
        if page.embedded_images:
            image_pages.append(page.page_number)
    meta = {
        "page_count": result.page_count,
        "content_types": [p.content_type for p in result.pages],
        "image_routed_pages": sorted(set(image_pages)),
    }
    return "\n\n".join(parts), meta


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
    p.add_argument("--pdf", type=Path, default=DEFAULT_PDF,
                   help="PDF to benchmark (default: the committed corpus doc)")
    p.add_argument("--out", type=Path, required=True, help="JSONL output path")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per model (default 3)")
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
        help="Seed for the run-order shuffle so the plan is reproducible",
    )
    args = p.parse_args(argv)

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    rng = random.Random(args.seed)

    client = ollama.Client(host=args.ollama_host or "http://localhost:11434")
    models = args.models or discover_models(client, include_coder=args.include_coder)
    if not models:
        raise SystemExit("No candidate models found (auto-discovery returned nothing).")

    print(f"Assembling summary input from {args.pdf.name} (pdfplumber + chunker)...",
          file=sys.stderr)
    source, meta = build_summary_input(args.pdf)
    if not source.strip():
        raise SystemExit(f"No extractable text in {args.pdf}")
    if meta["image_routed_pages"]:
        print(f"  WARNING: pages {meta['image_routed_pages']} would route to the "
              f"image/VLM pipeline — this doc is not purely image-free, so the input "
              f"omits content production would include.", file=sys.stderr)
    text, source_tokens = truncate_to_tokens(source, args.max_tokens)
    truncated = source_tokens > args.max_tokens

    plan = [(m, i) for m in models for i in range(args.runs)]
    rng.shuffle(plan)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")

    def write(record: dict):
        record["timestamp"] = time.time()
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

    write({
        "kind": "config",
        "pdf": str(args.pdf),
        "pdf_name": args.pdf.name,
        "page_count": meta["page_count"],
        "content_types": meta["content_types"],
        "image_routed_pages": meta["image_routed_pages"],
        "models": models,
        "runs_per_model": args.runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "source_token_count": source_tokens,
        "source_truncated": truncated,
        # The exact (truncated) text every model saw — kept so judge.py scores
        # against precisely this, no PDF re-extraction needed.
        "source_text": text,
        "shuffled_plan": [{"model": m, "run": i} for m, i in plan],
        "seed": args.seed,
    })

    print(f"Running {len(plan)} call(s): {len(models)} model(s) × {args.runs} run(s) "
          f"on {source_tokens} input token(s)...", file=sys.stderr)
    for idx, (model, run_idx) in enumerate(plan, 1):
        print(f"  [{idx}/{len(plan)}] {model} (run {run_idx})", file=sys.stderr, flush=True)
        result = call_summarize(client, model, text, args.temperature)
        write({"kind": "run", "model": model, "run_index": run_idx, **result})

    out_f.close()
    print(f"\nWrote results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
