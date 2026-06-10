#!/usr/bin/env python
"""Examine JSONL output from benchmarks/summarize_local.py.

Subcommands:
  timings     — per-model wall-clock + tok/s + schema gate
  blind       — write blinded summaries (one per model) + a key file
  errors      — list any failed runs, with raw-output previews
  leaderboard — merge timings + schema-valid % + judge scores into the final
                ranked report, dropping schema-failers and starring the frontier

`timings` and `leaderboard` render a colored rich table on a terminal and clean
markdown when piped/redirected — so `> leaderboard.md` stays diffable and free
of escape codes. Safe to run on a partially-written JSONL (a malformed final
line is skipped).

Examples:
  uv run python benchmarks/examine.py timings benchmarks/results.jsonl
  uv run python benchmarks/examine.py blind   benchmarks/results.jsonl --out benchmarks/blind/
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

from rich.console import Console
from rich.table import Table
from rich.text import Text


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


def _input_tokens_label(cfg: dict) -> str:
    """Token count the models actually saw, annotating truncation.

    A truncated run was fed exactly `--max-tokens`, so spell out the elision
    ("50,000 of 80,000 input tokens") rather than the full source count — which
    the original title overstated. Derived from the existing record fields, so
    pre-fix benchmark JSONL renders correctly too.
    """
    full = cfg.get("source_token_count")
    if full is None:
        return "? input tokens"
    if cfg.get("source_truncated"):
        return f"{cfg['max_tokens']:,} of {full:,} input tokens"
    return f"{full:,} input tokens"


def _runs_by_model(records: list[dict]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "run":
            by_model[r["model"]].append(r)
    return by_model


def _inference_seconds(run: dict) -> float:
    """Wall-clock minus model load time — the load-corrected inference cost."""
    return run["wall_seconds"] - (run.get("load_duration_ns") or 0) / 1e9


# ---- rendering: rich table on a TTY, markdown when piped ----

_MD_ALIGN = {"left": "---", "right": "--:", "center": ":-:"}


def _tty() -> bool:
    return sys.stdout.isatty()


def emit_table(title: str, columns: list[tuple[str, str]], rows: list[list[tuple[str, str | None]]]) -> None:
    """Render a table. ``columns`` is [(header, justify)]; each row cell is
    ``(text, style)`` where ``style`` is a rich style on a TTY and ignored in
    markdown. Numbers/colour on a terminal; portable markdown when redirected.
    An empty ``title`` is omitted (for tables that sit under their own heading)."""
    if _tty():
        table = Table(title=title or None, title_justify="left", header_style="bold")
        for header, justify in columns:
            table.add_column(header, justify=justify)
        for row in rows:
            table.add_row(*(Text(text, style=style or "") for text, style in row))
        Console().print(table)
    else:
        if title:
            print(f"# {title}\n")
        print("| " + " | ".join(h for h, _ in columns) + " |")
        print("|" + "|".join(_MD_ALIGN[j] for _, j in columns) + "|")
        for row in rows:
            print("| " + " | ".join(text for text, _ in row) + " |")


def note(line: str) -> None:
    """A footnote — dim on a TTY, italic in markdown."""
    if _tty():
        # highlight=False so rich doesn't recolor numbers/parens inside the note.
        Console(highlight=False).print(line, style="dim")
    else:
        print(f"\n_{line}_")


def heading(text: str, style: str = "bold") -> None:
    """A section heading — styled on a TTY, `## ` in markdown."""
    if _tty():
        Console(highlight=False).print(f"\n{text}", style=style)
    else:
        print(f"\n## {text}\n")


def cmd_timings(args) -> int:
    records = load_records(args.results)
    by_model = _runs_by_model(records)
    if not by_model:
        print("No measured runs yet.", file=sys.stderr)
        return 1

    cfg = _config(records)
    stats = []
    for model in sorted(by_model):
        rs = by_model[model]
        ok = [r for r in rs if r.get("ok")]
        infs = [_inference_seconds(r) for r in ok]
        tps = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second")]
        evals = [r["eval_count"] for r in ok if r.get("eval_count")]
        stats.append({
            "model": model, "n": len(rs), "ok": len(ok),
            "infs": infs, "tps": statistics.median(tps) if tps else None,
            "eval": statistics.median(evals) if evals else None,
        })
    fastest = max((s["tps"] for s in stats if s["tps"]), default=None)

    columns = [("Model", "left"), ("Schema", "left"), ("Inference (s)", "right"),
               ("Tok/s", "right"), ("Eval tok", "right")]
    rows = []
    for s in stats:
        passed = s["ok"] == s["n"] and s["ok"] > 0
        schema = (f"{'✓' if passed else '✗'} {s['ok']}/{s['n']}",
                  "green" if passed else "red")
        # Show the min–max range only when there are multiple runs AND it'd
        # render as nonzero; otherwise just the median (no "7.3 (7.3–7.3)").
        infs = s["infs"]
        if not infs:
            inf = ("—", None)
        elif len(infs) > 1 and round(max(infs) - min(infs), 1) > 0:
            inf = (f"{statistics.median(infs):.1f} ({min(infs):.1f}–{max(infs):.1f})", None)
        else:
            inf = (f"{statistics.median(infs):.1f}", None)
        tps = ((f"{s['tps']:.1f}", "bold green" if s["tps"] == fastest else None)
               if s["tps"] else ("—", None))
        ev = (str(int(s["eval"])), None) if s["eval"] else ("—", None)
        rows.append([(s["model"], None), schema, inf, tps, ev])

    title = (f"Timings · {cfg.get('pdf_name', '?')} · "
             f"{_input_tokens_label(cfg)} · "
             f"temp {cfg.get('temperature', '?')}")
    emit_table(title, columns, rows)
    note("Inference (s) = wall − load_duration (cold-start excluded); Tok/s from "
         "eval_duration. Schema = runs that parsed into DocumentSummary (the hard gate).")
    expected = cfg.get("runs_per_model")
    if expected and any(s["n"] < expected for s in stats):
        note(f"Expected {expected} runs per model; some are incomplete "
             f"(benchmark still running?).")
    missing = [m for m in cfg.get("models", []) if m not in by_model]
    if missing:
        note(f"Not yet seen: {', '.join(missing)}.")
    return 0


def cmd_blind(args) -> int:
    records = load_records(args.results)
    runs = [r for r in records if r.get("kind") == "run" and r.get("ok")]
    if not runs:
        print("No successful runs to blind yet.", file=sys.stderr)
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

    cfg = _config(records)
    with md_path.open("w") as f:
        f.write(f"# Blinded summaries — {cfg.get('pdf_name', '?')}\n\n")
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
        print(f"  {r['model']} (run {r.get('run_index')}): {r.get('error', '?')}")
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

    title = (f"Summarizer leaderboard · {cfg.get('pdf_name', '?')} · "
             f"{_input_tokens_label(cfg)} · "
             f"temp {cfg.get('temperature', '?')}")
    columns = [("Model", "left"), ("Schema-valid %", "right"), ("Tok/s", "right"),
               ("Inference (s)", "right"), ("Mean quality (/5)", "right"), ("Frontier", "center")]
    rows = []
    for s in survivors:
        on_front = s["model"] in front
        q = (f"{s['quality']:.2f}", None) if s["quality"] is not None else ("—", None)
        rows.append([
            (s["model"], "bold" if on_front else None),
            (f"{s['schema_pct']:.0f}%", None),
            (f"{s['tps']:.1f}", None),
            (f"{s['inference']:.1f}", None),
            q,
            ("★", "bold cyan") if on_front else ("", None),
        ])

    emit_table(title, columns, rows)
    if not args.judged:
        note("No judge scores supplied (--judged); quality column blank. "
             "Run benchmarks/judge.py to populate it.")
    note("Hard gate: a summarizer must return parseable DocumentSummary JSON every "
         "time; schema-failers are dropped below. ★ marks the speed/quality Pareto "
         "frontier among judged survivors (no other survivor is both faster and "
         "higher-quality).")

    if failers:
        heading(f"Disqualified — schema-valid % below {args.min_schema:.0f}%",
                style="bold red")
        note("These fail the hard structured-output gate and are unusable as "
             "summarizers regardless of speed.")
        emit_table("", [("Model", "left"), ("Schema-valid %", "right")],
                   [[(m, "red"), (f"{pct:.0f}%", "red")] for m, pct in sorted(failers)])
    return 0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("timings", help="Per-model timing + schema gate table")
    t.add_argument("results", type=Path)
    t.set_defaults(fn=cmd_timings)

    b = sub.add_parser("blind", help="Write blinded summaries + key file")
    b.add_argument("results", type=Path)
    b.add_argument("--out", type=Path, required=True, help="Output directory")
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
