"""`bartleby benchmark summarize` — run the (model × doc × runs) matrix.

Each call appends run records to the per-cell stores under
``benchmarks/results/``. The matrix order is shuffled so thermal throttling
and weight-eviction can't systematically favor any model; accumulation across
invocations replaces the old fixed batch — run it again whenever the machine
is idle and the evidence grows.

Providers:
- ``ollama:`` — the local client, streamed so the live view shows tokens
  accruing; timings come from Ollama's ``eval_count``/``eval_duration``.
- ``openai:`` — a structured-output chat completion validated against the
  same ``DocumentSummary``. Cloud models are *reference rows*: wall-clock is
  recorded but they carry no local-comparable throughput, run at provider
  default sampling (gpt-5.x rejects pinned temperatures), and sit off the
  leaderboard's Pareto frontier.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time

from bartleby.benchmark.progress import BenchmarkProgress
from bartleby.benchmark.refs import ModelRef, check_slug_collisions
from bartleby.benchmark.sources import (
    DEFAULT_EXTRACTION,
    DEFAULT_MAX_SUMMARIZE_TOKENS,
    ensure_source,
    load_corpus,
    select_documents,
)
from bartleby.benchmark.stores import (
    BenchmarkRoot,
    append_record,
    make_openai_client,
    read_records,
)
from bartleby.lib.consts import DEFAULT_TEMPERATURE


def prompt_sha() -> str:
    """Hash of the summarize prompt template + DocumentSummary schema, so a
    prompt or schema change shows up as a provenance break in the records."""
    from bartleby.providers.base import DocumentSummary
    from bartleby.providers.prompt import build_summary_messages

    canon = json.dumps(
        [build_summary_messages("{{DOCUMENT}}"), DocumentSummary.model_json_schema()],
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


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


def call_ollama(client, model: str, document_text: str, temperature: float,
                on_chunk=None) -> dict:
    """One streaming Ollama summarize call.

    Streams so ``on_chunk(content_so_far)`` can drive the live view; the final
    chunk carries the timing metadata. Accumulated content is validated against
    ``DocumentSummary`` at the end.
    """
    from pydantic import ValidationError

    from bartleby.providers.base import DocumentSummary
    from bartleby.providers.prompt import build_summary_messages

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
            # Whatever streamed before the failure — a timeout at 90% still
            # leaves forensics.
            "raw_output": content,
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


def call_openai(client, model: str, document_text: str) -> dict:
    """One cloud reference call via OpenAI structured output.

    No temperature is sent (gpt-5.x models reject pinned values) and no
    throughput is derived — wall-clock here measures a datacenter plus the
    network, not this machine.
    """
    from bartleby.providers.base import DocumentSummary
    from bartleby.providers.prompt import build_summary_messages

    wall_start = time.perf_counter()
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=build_summary_messages(document_text),
            response_format=DocumentSummary,
        )
    except Exception as e:
        return {"ok": False, "wall_seconds": time.perf_counter() - wall_start,
                "error": f"{type(e).__name__}: {e}"}
    wall_seconds = time.perf_counter() - wall_start
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return {"ok": False, "wall_seconds": wall_seconds,
                "error": "no parsed payload",
                "raw_output": response.choices[0].message.content or ""}
    usage = getattr(response, "usage", None)
    return {
        "ok": True,
        "wall_seconds": wall_seconds,
        "summary": parsed.model_dump(),
        "eval_count": getattr(usage, "completion_tokens", None),
        "prompt_eval_count": getattr(usage, "prompt_tokens", None),
    }


def _warn_missing_ollama(client, models: list[str]) -> None:
    """Surface listed-but-not-installed models up front; their runs will error."""
    installed = {m.model for m in client.list().models}
    missing = [m for m in models if m not in installed]
    if missing:
        print(f"WARNING: not installed (their runs will error — `ollama pull` "
              f"them or drop them from models.yaml): {', '.join(missing)}",
              file=sys.stderr)


def run(root: BenchmarkRoot, models: list[ModelRef] | None,
        documents: list[str] | None, runs: int = 1, seed: int | None = None,
        ollama_host: str | None = None,
        extraction: str = DEFAULT_EXTRACTION,
        ollama_client=None, openai_client=None) -> None:
    """Execute the matrix. ``ollama_client``/``openai_client`` are injectable
    for tests; by default they're constructed lazily per provider in use.

    ``extraction`` names the text-extraction backend whose cached source text
    (``sources/<doc-id>.txt`` for the default ``pdfplumber``, or
    ``sources/<doc-id>-<extraction>.txt`` for named variants) the models
    receive. Named variants must already exist as pre-committed fixtures or a
    prior extraction run; they are never produced live here.
    """
    root.require()
    refs = models if models is not None else root.load_models()
    check_slug_collisions(refs)
    cc = [r for r in refs if r.provider == "anthropic-cc"]
    if cc:
        raise SystemExit(
            f"anthropic-cc is a judge-only provider, not a summarize backend "
            f"(got {', '.join(str(r) for r in cc)}). Use it with "
            f"`bartleby benchmark judge --model anthropic-cc/<model>`.")
    corpus = select_documents(load_corpus(root), documents)
    sources = {doc_id: ensure_source(root, doc_id, pdf, extraction=extraction)
               for doc_id, pdf in corpus.items()}

    if ollama_client is None and any(r.provider == "ollama" for r in refs):
        import ollama
        ollama_client = ollama.Client(host=ollama_host or "http://localhost:11434")
    if ollama_client is not None:
        _warn_missing_ollama(
            ollama_client, [r.model for r in refs if r.provider == "ollama"])
    if openai_client is None and any(r.provider == "openai" for r in refs):
        openai_client = make_openai_client(root)

    plan = [(ref, doc_id) for ref in refs for doc_id in corpus
            for _ in range(runs)]
    random.Random(seed).shuffle(plan)

    # Per-cell run_index continues from whatever the store already holds.
    next_index = {
        (ref, doc_id): len(read_records(root.result_path(ref, doc_id, extraction)))
        for ref, doc_id in set(plan)
    }

    psha = prompt_sha()
    from bartleby import __version__

    print(f"Running {len(plan)} call(s): {len(refs)} model(s) × "
          f"{len(corpus)} doc(s) × {runs} run(s) "
          f"[extraction={extraction}]...", file=sys.stderr)
    with BenchmarkProgress([str(r) for r in refs], len(corpus) * runs,
                           len(plan)) as prog:
        for call_no, (ref, doc_id) in enumerate(plan, 1):
            run_index = next_index[(ref, doc_id)]
            next_index[(ref, doc_id)] += 1
            prog.start_call(str(ref), doc_id, run_index, call_no)
            if ref.provider == "ollama":
                result = call_ollama(ollama_client, ref.model,
                                     sources[doc_id].text, DEFAULT_TEMPERATURE,
                                     on_chunk=prog.on_chunk)
                temperature = DEFAULT_TEMPERATURE
            else:
                result = call_openai(openai_client, ref.model, sources[doc_id].text)
                temperature = None  # provider default; see call_openai
            prog.finish_call(str(ref), result.get("ok", False),
                             result.get("tokens_per_second"))
            append_record(root.result_path(ref, doc_id, extraction), {
                "provider": ref.provider,
                "model": ref.model,
                "doc": doc_id,
                "extraction": extraction,
                "run_index": run_index,
                **result,
                "source_sha": sources[doc_id].sha,
                "source_tokens": sources[doc_id].tokens,
                "prompt_sha": psha,
                "temperature": temperature,
                "max_tokens": DEFAULT_MAX_SUMMARIZE_TOKENS,
                "bartleby_version": __version__,
            })

    print(f"\nAppended {len(plan)} record(s) under {root.results_dir}/",
          file=sys.stderr)
