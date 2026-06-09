#!/usr/bin/env python
"""Benchmark local Ollama models against Bartleby's summarization prompt.

Benchmarks one committed PDF (``benchmarks/corpus/``), assembling its
summarization input **the same way production does**: extract text with the
pdfplumber backend, chunk it with Bartleby's chunker, and join the document
chunks in page/chunk order — exactly what ``writer.summary_input`` produces for
an image-free document. Then each model in ``benchmarks/models.yaml``
summarizes that input N times, at production settings (temp 0.0, the default
``max_summarize_tokens``), using Bartleby's actual ``build_summary_messages`` +
``DocumentSummary`` schema.

The model set is **exactly** ``models.yaml`` (or ``--models``) — nothing is
auto-discovered or skipped; curating the list is the human's job.

The (model, run) matrix is shuffled into one randomized order so thermal
throttling and weight-eviction can't systematically favor any model. Calls
stream so the live progress view can show tokens accruing in real time.

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
import yaml
from pydantic import ValidationError
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

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

DEFAULT_MODELS_FILE = Path(__file__).parent / "models.yaml"


def load_models(models_file: Path, override: list[str] | None) -> list[str]:
    """The exact model list to run — ``--models`` if given, else the YAML's."""
    if override:
        return override
    if not models_file.exists():
        raise SystemExit(f"Models file not found: {models_file} (or pass --models)")
    data = yaml.safe_load(models_file.read_text()) or {}
    models = data.get("models") or []
    if not models:
        raise SystemExit(f"No models listed under `models:` in {models_file}")
    return list(models)


def warn_missing(client: ollama.Client, models: list[str]) -> None:
    """Surface listed-but-not-installed models up front; their runs will error."""
    installed = {m.model for m in client.list().models}
    missing = [m for m in models if m not in installed]
    if missing:
        print(f"WARNING: not installed (their runs will error — `ollama pull` "
              f"them or drop them from models.yaml): {', '.join(missing)}",
              file=sys.stderr)


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


_ENCODING = None


def _cl100k():
    """Lazy, process-wide cl100k_base encoder (shared by truncation + live count)."""
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    tok = _cl100k()
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
    on_chunk=None,
) -> dict:
    """One streaming Ollama summarize call.

    Streams so ``on_chunk(content_so_far)`` can drive a live view; the final
    chunk carries the timing metadata. Accumulated content is validated against
    ``DocumentSummary`` at the end, exactly as the non-streaming path was.
    """
    wall_start = time.perf_counter()
    content = ""
    final = None
    try:
        for chunk in client.chat(
            model=model,
            messages=build_summary_messages(document_text),
            format=DocumentSummary.model_json_schema(),
            options={"temperature": temperature},
            stream=True,
        ):
            content += chunk.message.content or ""
            if on_chunk is not None:
                on_chunk(content)
            if getattr(chunk, "done", False):
                final = chunk
    except Exception as e:
        return {
            "ok": False,
            "wall_seconds": time.perf_counter() - wall_start,
            "error": f"{type(e).__name__}: {e}",
        }
    wall_seconds = time.perf_counter() - wall_start

    timings = _extract_timings(final) if final is not None else {}
    try:
        summary = DocumentSummary.model_validate_json(content)
    except ValidationError as e:
        return {
            "ok": False,
            "wall_seconds": wall_seconds,
            "error": f"schema validation failed: {e}",
            "raw_output": content,
            **timings,
        }
    return {
        "ok": True,
        "wall_seconds": wall_seconds,
        "summary": summary.model_dump(),
        **timings,
    }


class BenchmarkProgress:
    """Live rich view — overall matrix bar + streaming active-model line + a
    pulse bar + a per-model dashboard table. Off a TTY (piped/CI) it degrades
    to plain stderr lines so logs stay readable.

    The object is its own renderable (``__rich__``); Live re-reads its mutable
    state on each refresh tick, so per-chunk callbacks only stash the latest
    text and the (throttled) token estimate is computed at render time, not on
    every one of the hundreds of stream chunks.
    """

    def __init__(self, models: list[str], runs: int, total_calls: int):
        # isatty() is the honest signal — rich's is_terminal over-detects when
        # FORCE_COLOR/CI-style env vars are set, which would spray escape codes
        # into a redirected log. Animate only on a real terminal.
        self.tty = sys.stderr.isatty()
        self.console = Console(file=sys.stderr, force_terminal=self.tty)
        self.runs = runs
        self.total = total_calls
        self.state = {m: {"n": 0, "tps": None, "passed": None, "running": False}
                      for m in models}
        self.active: dict | None = None

        self._overall = Progress(
            TextColumn("[bold]Summarizing"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(), TextColumn("calls"),
            TimeElapsedColumn(), TextColumn("• ETA"), TimeRemainingColumn(),
            console=self.console,
        )
        self._task = self._overall.add_task("calls", total=total_calls)
        self._pulse = Progress(
            TextColumn("   "),
            BarColumn(bar_width=40, pulse_style="cyan"),
            TextColumn("[dim]generating…"),
            console=self.console,
        )
        self._pulse.add_task("gen", total=None)  # total=None -> animated pulse
        self._live = Live(self, console=self.console, refresh_per_second=12)

    # -- rich integration --
    def __rich__(self):
        parts = [self._overall]
        if self.active is not None:
            parts += [self._active_line(), self._pulse]
        parts.append(self._table())
        return Group(*parts)

    def _est_tokens(self, content: str) -> int:
        return len(_cl100k().encode(content)) if content else 0

    def _active_line(self) -> Text:
        # Live estimate: tiktoken on the streamed text so far / wall time. This
        # is intentionally distinct from the table's Tok/s (Ollama's exact
        # eval_count / eval_duration, known only at completion) — this one shows
        # "tokens accruing now", that one shows measured throughput.
        a = self.active
        elapsed = time.perf_counter() - a["start"]
        toks = self._est_tokens(a["content"])
        tps = toks / elapsed if elapsed > 0 else 0
        return Text.assemble(
            ("▶ ", "cyan"),
            (f"{a['model']} (run {a['run']})", "bold"),
            (f"   ~{toks} tok · {tps:.0f} tok/s · {elapsed:.1f}s", "dim"),
        )

    def _table(self) -> Table:
        t = Table(box=None, pad_edge=False, expand=False)
        t.add_column("Model")
        t.add_column("Status")
        t.add_column("Runs", justify="right")
        t.add_column("Tok/s", justify="right")
        t.add_column("OK")  # all runs so far succeeded (parsed DocumentSummary)?
        for model, st in self.state.items():
            if st["running"]:
                status = Text("▶ running", style="cyan")
            elif st["n"] >= self.runs:
                status = Text("✓ done", style="green")
            elif st["n"]:
                status = Text("waiting", style="yellow")
            else:
                status = Text("queued", style="dim")
            tps = f"{st['tps']:.1f}" if st["tps"] else "—"
            if st["passed"] is None:
                ok_cell = Text("—", style="dim")
            elif st["passed"]:
                ok_cell = Text("ok", style="green")
            else:
                ok_cell = Text("FAIL", style="red")
            t.add_row(model, status, f"{st['n']}/{self.runs}", tps, ok_cell)
        return t

    # -- lifecycle --
    def __enter__(self):
        if self.tty:
            self._live.start()
        return self

    def __exit__(self, *exc):
        if self.tty:
            self._live.refresh()
            self._live.stop()

    def start_call(self, model: str, run_idx: int, call_no: int) -> None:
        self.active = {"model": model, "run": run_idx, "content": "",
                       "start": time.perf_counter()}
        self.state[model]["running"] = True
        if not self.tty:
            print(f"  [{call_no}/{self.total}] {model} (run {run_idx})",
                  file=sys.stderr, flush=True)

    def on_chunk(self, content: str) -> None:
        if self.active is not None:
            self.active["content"] = content  # render reads this on its own tick

    def finish_call(self, model: str, ok: bool, tps: float | None) -> None:
        st = self.state[model]
        st["running"] = False
        st["n"] += 1
        st["passed"] = ok if st["passed"] is None else (st["passed"] and ok)
        if ok and tps:
            st["tps"] = tps
        self.active = None
        if self.tty:
            self._overall.advance(self._task)
        else:
            extra = f" · {tps:.0f} tok/s" if (ok and tps) else ""
            print(f"      {'✓' if ok else '✗'} {model}{extra}",
                  file=sys.stderr, flush=True)


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
        "--models-file", type=Path, default=DEFAULT_MODELS_FILE,
        help="YAML listing the models to run (default benchmarks/models.yaml)",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Ad-hoc model list, overrides --models-file (no filtering either way)",
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
    models = load_models(args.models_file, args.models)
    warn_missing(client, models)

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
    with BenchmarkProgress(models, args.runs, len(plan)) as prog:
        for call_no, (model, run_idx) in enumerate(plan, 1):
            prog.start_call(model, run_idx, call_no)
            result = call_summarize(client, model, text, args.temperature,
                                    on_chunk=prog.on_chunk)
            prog.finish_call(model, result.get("ok", False),
                             result.get("tokens_per_second"))
            write({"kind": "run", "model": model, "run_index": run_idx, **result})

    out_f.close()
    print(f"\nWrote results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
