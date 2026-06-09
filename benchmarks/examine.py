#!/usr/bin/env python
"""Examine JSONL output from benchmarks/summarize_local.py.

Subcommands:
  timings     — markdown table of per-model wall-clock + tok/s + schema-valid %
  blind       — write blinded summaries for one doc (one per model) + a key file
  errors      — list any failed runs, with raw-output previews
  leaderboard — merge timings + schema-valid % + judge scores into the final
                ranked report, dropping schema-failers and starring the frontier

Safe to run on a partially-written JSONL (malformed final line is skipped).

Examples:
  uv run python benchmarks/examine.py timings benchmarks/results.jsonl
  uv run python benchmarks/examine.py blind   benchmarks/results.jsonl --document-id 45 --out benchmarks/blind/
  uv run python benchmarks/examine.py errors  benchmarks/results.jsonl
  uv run python benchmarks/examine.py leaderboard benchmarks/results.jsonl --judged benchmarks/judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Final line may be partial if the benchmark is still writing.
                print(f"warning: skipping malformed line in {path}", file=sys.stderr)
    return records


def _config(records: list[dict]) -> dict:
    return next((r for r in records if r.get("kind") == "config"), {})


def _runs_by_model(records: list[dict]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "run":
            by_model[r["model"]].append(r)
    return by_model


def _inference_seconds(run: dict) -> float:
    """Wall-clock minus model load time — the comparable inference cost."""
    return run["wall_seconds"] - (run.get("load_duration_ns") or 0) / 1e9


def cmd_timings(args) -> int:
    records = load_records(args.results)
    by_model = _runs_by_model(records)
    if not by_model:
        print("No measured runs yet.", file=sys.stderr)
        return 1

    cfg = _config(records)
    n_docs = len(cfg.get("document_ids", []))
    print(f"# Timings — {n_docs} doc(s), {len(cfg.get('models', []))} model(s) "
          f"(temperature={cfg.get('temperature', '?')}, "
          f"max_tokens={cfg.get('max_tokens', '?')})\n")
    print("| Model | OK / N | Schema-valid % | Inference sec (median) | Inference range | Tok/s (median) | Eval tokens (median) |")
    print("|---|---|---|---|---|---|---|")

    for model in sorted(by_model):
        rs = by_model[model]
        ok = [r for r in rs if r.get("ok")]
        n_str = f"{len(ok)} / {len(rs)}"
        valid_pct = f"{100 * len(ok) / len(rs):.0f}%" if rs else "—"
        if not ok:
            print(f"| {model} | {n_str} | {valid_pct} | — | — | — | — |")
            continue
        infs = [_inference_seconds(r) for r in ok]
        tps = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second")]
        evals = [r["eval_count"] for r in ok if r.get("eval_count")]
        tps_med = f"{statistics.median(tps):.1f}" if tps else "—"
        eval_med = f"{int(statistics.median(evals))}" if evals else "—"
        print(f"| {model} | {n_str} | {valid_pct} | {statistics.median(infs):.1f}s | "
              f"{min(infs):.1f}–{max(infs):.1f}s | {tps_med} | {eval_med} |")

    print("\n_Inference sec = wall_seconds − load_duration. "
          "Schema-valid % = share of runs that parsed into DocumentSummary (the hard gate)._")
    expected = cfg.get("runs_per_model")
    if expected and n_docs:
        full = expected * n_docs
        if any(len(by_model[m]) < full for m in by_model):
            print(f"\n_Note: expected {full} runs per model ({expected} × {n_docs} docs); "
                  f"some are incomplete (benchmark still running?)._")
    missing = [m for m in cfg.get("models", []) if m not in by_model]
    if missing:
        print(f"\n_Not yet seen: {', '.join(missing)}._")
    return 0


def cmd_blind(args) -> int:
    records = load_records(args.results)
    cfg = _config(records)
    doc_ids = cfg.get("document_ids", [])
    document_id = args.document_id or (doc_ids[0] if doc_ids else None)
    runs = [r for r in records
            if r.get("kind") == "run" and r.get("ok")
            and r.get("document_id") == document_id]
    if not runs:
        print(f"No successful runs to blind for document {document_id}.", file=sys.stderr)
        return 1

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_model[r["model"]].append(r)

    rng = random.Random(args.seed)

    # Pick one run per model (random if multiple available); then shuffle the
    # (model, run) list so label order doesn't leak the alphabetic model order.
    chosen = [(model, rng.choice(by_model[model])) for model in sorted(by_model)]
    rng.shuffle(chosen)
    labels = [chr(ord("A") + i) for i in range(len(chosen))]

    args.out.mkdir(parents=True, exist_ok=True)
    md_path = args.out / "blinded_summaries.md"
    key_path = args.out / "key.json"

    with md_path.open("w") as f:
        f.write(f"# Blinded summaries — doc {document_id}\n\n")
        f.write(f"_{len(chosen)} models, 1 run each (picked randomly from completed runs)._\n\n")
        for label, (_, run) in zip(labels, chosen):
            s = run["summary"]
            title_len = len(s["title"])
            desc_words = len(s["description"].split())
            text_words = len(s["text"].split())
            f.write(f"## {label}\n\n")
            f.write(f"_title {title_len} chars · description {desc_words} words · text {text_words} words_\n\n")
            f.write(f"**title**: {s['title']}\n\n")
            f.write(f"**description**: {s['description']}\n\n")
            f.write(f"**text**:\n\n{s['text']}\n\n---\n\n")

    key = {label: model for label, (model, _) in zip(labels, chosen)}
    with key_path.open("w") as f:
        json.dump(key, f, indent=2)

    print(f"Wrote {md_path}")
    print(f"Wrote {key_path}  (keep this aside until judging is done)")
    print(f"\nLabels assigned: {' '.join(labels)}")
    return 0


def cmd_errors(args) -> int:
    records = load_records(args.results)
    run_fails = [r for r in records if r.get("kind") == "run" and not r.get("ok")]

    if not run_fails:
        print("No errors.")
        return 0

    print(f"=== Run failures ({len(run_fails)}) ===")
    for r in run_fails:
        print(f"  {r['model']} · doc {r.get('document_id')} (run {r.get('run_index')}): "
              f"{r.get('error', '?')}")
        if r.get("raw_output"):
            print(f"    raw: {r['raw_output'][:200]}...")
    return 0


def _quality_by_model(judged_path: Path) -> dict[str, float]:
    """Mean overall quality per model from a judge JSONL (kind=judgment, ok)."""
    scores: dict[str, list[float]] = defaultdict(list)
    for r in load_records(judged_path):
        if r.get("kind") == "judgment" and r.get("ok"):
            scores[r["model"]].append(r["scores"]["mean"])
    return {m: statistics.mean(v) for m, v in scores.items() if v}


def _frontier(rows: list[dict]) -> set[str]:
    """Models on the speed/quality Pareto frontier: none is both faster AND better."""
    front: set[str] = set()
    for a in rows:
        dominated = any(
            b["model"] != a["model"]
            and b["tps"] >= a["tps"] and b["quality"] >= a["quality"]
            and (b["tps"] > a["tps"] or b["quality"] > a["quality"])
            for b in rows
        )
        if not dominated:
            front.add(a["model"])
    return front


def cmd_leaderboard(args) -> int:
    records = load_records(args.results)
    by_model = _runs_by_model(records)
    if not by_model:
        print("No measured runs yet.", file=sys.stderr)
        return 1

    quality = _quality_by_model(args.judged) if args.judged else {}
    cfg = _config(records)

    survivors: list[dict] = []
    failers: list[tuple[str, float]] = []
    for model in sorted(by_model):
        rs = by_model[model]
        ok = [r for r in rs if r.get("ok")]
        valid_pct = 100 * len(ok) / len(rs) if rs else 0.0
        if valid_pct < args.min_schema or not ok:
            failers.append((model, valid_pct))
            continue
        tps = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second")]
        survivors.append({
            "model": model,
            "schema_pct": valid_pct,
            "tps": statistics.median(tps) if tps else 0.0,
            "inference": statistics.median([_inference_seconds(r) for r in ok]),
            "quality": quality.get(model),
        })

    # The frontier needs a quality value, so models lacking a judge score sit
    # off it; sort the table by quality (when judged) then speed.
    rankable = [s for s in survivors if s["quality"] is not None]
    front = _frontier(rankable) if rankable else set()
    survivors.sort(key=lambda s: (-(s["quality"] or -1), -s["tps"]))

    print(f"# Summarizer leaderboard — {len(cfg.get('document_ids', []))} doc(s), "
          f"temp {cfg.get('temperature', '?')}, max_tokens {cfg.get('max_tokens', '?')}\n")
    if not args.judged:
        print("_No judge scores supplied (--judged); quality column blank. "
              "Run benchmarks/judge.py to populate it._\n")
    print("| Model | Schema-valid % | Tok/s (median) | Inference sec (median) | Mean quality (/5) | Frontier |")
    print("|---|---|---|---|---|---|")
    for s in survivors:
        q = f"{s['quality']:.2f}" if s["quality"] is not None else "—"
        star = "★" if s["model"] in front else ""
        print(f"| {s['model']} | {s['schema_pct']:.0f}% | {s['tps']:.1f} | "
              f"{s['inference']:.1f}s | {q} | {star} |")

    if failers:
        print(f"\n## Disqualified — schema-valid % below {args.min_schema:.0f}%\n")
        print("These fail the hard structured-output gate and are unusable as "
              "summarizers regardless of speed:\n")
        for model, pct in sorted(failers):
            print(f"- **{model}** — {pct:.0f}% schema-valid")

    print("\n_Hard gate: a summarizer must return parseable DocumentSummary JSON every "
          "time; schema-failers are dropped above. ★ marks the speed/quality Pareto "
          "frontier among judged survivors (no other survivor is both faster and "
          "higher-quality)._")
    return 0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("timings", help="Print a markdown timing + schema-valid table")
    t.add_argument("results", type=Path)
    t.set_defaults(fn=cmd_timings)

    b = sub.add_parser("blind", help="Write blinded summaries + key file for one doc")
    b.add_argument("results", type=Path)
    b.add_argument("--out", type=Path, required=True, help="Output directory")
    b.add_argument("--document-id", type=int, default=None,
                   help="Which document to blind (default: first sampled doc)")
    b.add_argument("--seed", type=int, default=None,
                   help="Seed the per-model pick + label shuffle for reproducibility")
    b.set_defaults(fn=cmd_blind)

    e = sub.add_parser("errors", help="List failed runs")
    e.add_argument("results", type=Path)
    e.set_defaults(fn=cmd_errors)

    lb = sub.add_parser("leaderboard", help="Merge timings + schema %% + judge scores")
    lb.add_argument("results", type=Path)
    lb.add_argument("--judged", type=Path, default=None,
                    help="Judge JSONL from benchmarks/judge.py (adds the quality column)")
    lb.add_argument("--min-schema", type=float, default=100.0,
                    help="Minimum schema-valid %% to survive the hard gate (default 100)")
    lb.set_defaults(fn=cmd_leaderboard)

    args = p.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
