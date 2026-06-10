"""`bartleby benchmark leaderboard` / `blind` / `errors` — read the stores.

Aggregation: judge passes over the same distinct summary average into one
per-summary score; distinct summaries combine weighted by how many OK runs
produced each (recomputed from the results store at read time — judgment
records go stale as runs accumulate); docs weight equally. The Pareto
frontier covers local providers only — a cloud reference row's wall-clock
measures someone else's datacenter, so it anchors the quality axis and
nothing else.

Tables render rich on a TTY and clean markdown when piped; ``--output``
additionally writes the leaderboard as CSV.
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from bartleby.benchmark.judging import summary_sha
from bartleby.benchmark.refs import ModelRef
from bartleby.benchmark.stores import BenchmarkRoot, in_window, read_store


# ---- loading ----

def _ref(record: dict) -> ModelRef:
    return ModelRef(record["provider"], record["model"])


def load_runs(root: BenchmarkRoot, models: list[ModelRef] | None = None,
              documents: list[str] | None = None,
              since: float | None = None, until: float | None = None) -> list[dict]:
    runs = read_store(root.results_dir)
    return [r for r in runs
            if in_window(r, since, until)
            and (models is None or _ref(r) in models)
            and (documents is None or r["doc"] in documents)]


def load_judgments(root: BenchmarkRoot, judges: list[ModelRef] | None = None,
                   models: list[ModelRef] | None = None,
                   documents: list[str] | None = None,
                   since: float | None = None, until: float | None = None) -> list[dict]:
    judgments = read_store(root.judgements_dir)
    return [j for j in judgments
            if in_window(j, since, until)
            and (judges is None or
                 ModelRef(j["judge_provider"], j["judge_model"]) in judges)
            and (models is None or _ref(j) in models)
            and (documents is None or j["doc"] in documents)]


# ---- aggregation ----

def quality_cells(runs: list[dict], judgments: list[dict]) -> dict[tuple, dict]:
    """Per (ref, doc): the weighted quality score plus its evidence counts.

    Weights come from the windowed runs (OK run count per summary_sha), never
    from judgment records. Judged summaries no run in the window produced get
    weight 0 and drop out; unjudged summaries are visible as evidence counts
    but can't contribute a score.
    """
    # (ref, doc) -> sha -> ok-run count
    run_counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in runs:
        if r.get("ok"):
            run_counts[(_ref(r), r["doc"])][summary_sha(r["summary"])] += 1

    # (ref, doc) -> sha -> [pass means] (pooled across selected judges)
    pass_means: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        if j.get("ok"):
            pass_means[(_ref(j), j["doc"])][j["summary_sha"]].append(j["scores"]["mean"])

    cells: dict[tuple, dict] = {}
    for cell, shas in run_counts.items():
        judged = {sha: means for sha, means in pass_means.get(cell, {}).items()
                  if sha in shas}
        weight = sum(shas[sha] for sha in judged)
        score = (sum(statistics.mean(means) * shas[sha]
                     for sha, means in judged.items()) / weight
                 if weight else None)
        cells[cell] = {
            "score": score,
            "runs": sum(shas.values()),
            "distinct": len(shas),
            "judged": len(judged),
            "passes": sum(len(m) for m in judged.values()),
        }
    return cells


def quality_by_ref(cells: dict[tuple, dict]) -> dict[ModelRef, float]:
    """Overall quality per model: docs weight equally, regardless of run counts."""
    by_ref: dict[ModelRef, list[float]] = defaultdict(list)
    for (ref, _), cell in cells.items():
        if cell["score"] is not None:
            by_ref[ref].append(cell["score"])
    return {ref: statistics.mean(scores) for ref, scores in by_ref.items()}


def _inference_seconds(run: dict) -> float:
    """Wall-clock minus model load time — the load-corrected inference cost."""
    return run["wall_seconds"] - (run.get("load_duration_ns") or 0) / 1e9


def _frontier(rows: list[dict]) -> set[ModelRef]:
    """Local models on the speed/quality Pareto frontier: none is both faster
    AND better. Cloud reference rows never qualify (their wall-clock measures
    someone else's datacenter)."""
    local = [r for r in rows if r["ref"].local and r["quality"] is not None
             and r["input_tps"] is not None]
    front: set[ModelRef] = set()
    for a in local:
        dominated = any(
            b["ref"] != a["ref"]
            and b["input_tps"] >= a["input_tps"] and b["quality"] >= a["quality"]
            and (b["input_tps"] > a["input_tps"] or b["quality"] > a["quality"])
            for b in local
        )
        if not dominated:
            front.add(a["ref"])
    return front


# The provenance axes a cell must hold constant for its scores to average
# cleanly. temperature is None for cloud reference rows (provider default) —
# constant within a cell, since a cell never mixes providers.
_REGIME_KEYS = ("source_sha", "prompt_sha", "temperature", "max_tokens")


def heterogeneity_warnings(runs: list[dict]) -> list[str]:
    """Cells whose windowed runs mix provenance regimes — averaging across
    them silently would compare apples to drifted oranges."""
    mixes: dict[tuple, dict[str, set]] = defaultdict(
        lambda: {key: set() for key in _REGIME_KEYS})
    for r in runs:
        cell = mixes[(_ref(r), r["doc"])]
        for key in _REGIME_KEYS:
            if r.get(key) is not None:
                cell[key].add(r[key])
    warnings = []
    for (ref, doc), shas in sorted(mixes.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        mixed = [key for key, vals in shas.items() if len(vals) > 1]
        if mixed:
            warnings.append(f"{ref} · {doc} mixes {' and '.join(mixed)} values "
                            f"within this window — scores average across regimes.")
    return warnings


# ---- rendering: rich table on a TTY, markdown when piped ----

_MD_ALIGN = {"left": "---", "right": "--:", "center": ":-:"}


def _tty() -> bool:
    return sys.stdout.isatty()


def emit_table(title: str, columns: list[tuple[str, str]],
               rows: list[list[tuple[str, str | None]]]) -> None:
    """``columns`` is [(header, justify)]; each row cell is ``(text, style)``
    where ``style`` is a rich style on a TTY and ignored in markdown. An empty
    ``title`` is omitted (for tables under their own heading)."""
    if _tty():
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

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
        from rich.console import Console

        # highlight=False so rich doesn't recolor numbers/parens inside the note.
        Console(highlight=False).print(line, style="dim")
    else:
        print(f"\n_{line}_")


def heading(text: str, style: str = "bold") -> None:
    """A section heading — styled on a TTY, `## ` in markdown."""
    if _tty():
        from rich.console import Console

        Console(highlight=False).print(f"\n{text}", style=style)
    else:
        print(f"\n## {text}\n")


# ---- subcommands ----

def leaderboard(root: BenchmarkRoot, models: list[ModelRef] | None = None,
                documents: list[str] | None = None,
                judges: list[ModelRef] | None = None,
                since: float | None = None, until: float | None = None,
                output: Path | None = None, min_schema: float = 100.0) -> int:
    root.require()
    runs = load_runs(root, models, documents, since, until)
    if not runs:
        print("No runs on record (in this window).", file=sys.stderr)
        return 1
    judgments = load_judgments(root, judges, models, documents, since, until)
    cells = quality_cells(runs, judgments)
    quality = quality_by_ref(cells)
    docs = sorted({r["doc"] for r in runs})

    by_ref: dict[ModelRef, list[dict]] = defaultdict(list)
    for r in runs:
        by_ref[_ref(r)].append(r)

    survivors: list[dict] = []
    failers: list[tuple[ModelRef, float]] = []
    for ref in sorted(by_ref, key=str):
        rs = by_ref[ref]
        ok = [r for r in rs if r.get("ok")]
        valid_pct = 100 * len(ok) / len(rs)
        if valid_pct < min_schema or not ok:
            failers.append((ref, valid_pct))
            continue
        # Input tokens/s = prompt tokens / load-corrected wall-clock —
        # deliberately charges generation time against *input* length (the
        # rationale lives in benchmarks/README.md and the GH-0303 hotfix
        # decision doc). prompt_eval_count is recorded for every provider, so
        # cloud rows get a number too. Mean of per-doc medians: a flat median
        # across docs would track the corpus mix.
        per_doc_itps: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            secs = _inference_seconds(r)
            if r.get("prompt_eval_count") and secs > 0:
                per_doc_itps[r["doc"]].append(r["prompt_eval_count"] / secs)
        survivors.append({
            "ref": ref,
            "schema_pct": valid_pct,
            "input_tps": statistics.mean(
                [statistics.median(v) for v in per_doc_itps.values()])
            if per_doc_itps else None,
            "quality": quality.get(ref),
            "runs": len(ok),
        })

    front = _frontier(survivors)
    survivors.sort(key=lambda s: (-(s["quality"] if s["quality"] is not None else -1),
                                  -(s["input_tps"] or 0)))

    window = ""
    if since or until:
        window = " · windowed"
    columns = [("Model", "left"), ("Schema-valid %", "right"),
               ("Input tokens/s", "right"), ("Mean quality (/5)", "right"),
               ("Runs", "right"), ("Pareto optimal", "center")]
    rows = []
    for s in survivors:
        on_front = s["ref"] in front
        name = str(s["ref"]) + ("" if s["ref"].local else " †")
        rows.append([
            (name, "bold" if on_front else None),
            (f"{s['schema_pct']:.0f}%", None),
            (f"{s['input_tps']:.1f}" if s["input_tps"] is not None else "—", None),
            (f"{s['quality']:.2f}" if s["quality"] is not None else "—", None),
            (str(s["runs"]), None),
            ("★", "bold cyan") if on_front else ("", None),
        ])
    emit_table(f"Summarizer leaderboard · {len(docs)} doc(s){window}", columns, rows)
    if not judgments:
        note("No judgments on record (in this window); quality column blank. "
             "Run `bartleby benchmark judge` to populate it.")
    note("Hard gate: a summarizer must return parseable DocumentSummary JSON every "
         "time; schema-failers are dropped below. ★ marks the speed/quality Pareto "
         "frontier among judged local survivors — those no other local model beats "
         "on both axes. † cloud reference row: wall-clock isn't comparable, never on "
         "the frontier. Input tokens/s = source-document tokens per load-corrected "
         "wall-clock second (mean of per-doc medians): divide a document's token "
         "count by it to estimate its summarization time. Measured for † rows too, "
         "though there it still counts the datacenter and network. quality = mean "
         "over docs of run-weighted, pass-averaged judge scores (a model missing a "
         "doc's judgments averages over fewer docs).")
    for warning in heterogeneity_warnings(runs):
        note(f"WARNING: {warning}")

    if any(cell["score"] is not None for cell in cells.values()):
        heading("Mean quality by document")
        note("Cell = score (OK runs / judge passes); — = no judgments yet.")
        doc_cols = [(d, "right") for d in docs]
        q_rows = []
        for s in survivors:
            row = [(str(s["ref"]), None)]
            for d in docs:
                cell = cells.get((s["ref"], d))
                if cell is None:
                    row.append(("—", None))
                elif cell["score"] is None:
                    row.append((f"— ({cell['runs']}r/0p)", None))
                else:
                    row.append((f"{cell['score']:.2f} ({cell['runs']}r/{cell['passes']}p)", None))
            q_rows.append(row)
        emit_table("", [("Model", "left"), *doc_cols], q_rows)

    if failers:
        heading(f"Disqualified — schema-valid % below {min_schema:.0f}%",
                style="bold red")
        note("These fail the hard structured-output gate and are unusable as "
             "summarizers regardless of speed. (A model that never produced an "
             "OK run — e.g. it failed to load — also lands here; check `errors`.)")
        emit_table("", [("Model", "left"), ("Schema-valid %", "right")],
                   [[(str(ref), "red"), (f"{pct:.0f}%", "red")]
                    for ref, pct in failers])

    if output is not None:
        _write_csv(output, survivors, front, cells, docs)
        print(f"Wrote {output}", file=sys.stderr)
    return 0


def _write_csv(path: Path, survivors: list[dict], front: set[ModelRef],
               cells: dict[tuple, dict], docs: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "model", "schema_valid_pct",
                         "input_tokens_per_second", "mean_quality", "runs",
                         "pareto_optimal", *[f"quality_{d}" for d in docs]])
        for s in survivors:
            per_doc = [(cells.get((s["ref"], d)) or {}).get("score") for d in docs]
            writer.writerow([
                s["ref"].provider, s["ref"].model, round(s["schema_pct"], 1),
                round(s["input_tps"], 2) if s["input_tps"] is not None else None,
                s["quality"], s["runs"], s["ref"] in front,
                *per_doc,
            ])


def blind(root: BenchmarkRoot, out_dir: Path, seed: int | None = None,
          models: list[ModelRef] | None = None,
          documents: list[str] | None = None) -> int:
    """Blinded summaries + key file for the human spot-check of the judge."""
    import itertools
    import json
    import string

    root.require()
    runs = [r for r in load_runs(root, models, documents) if r.get("ok")]
    if not runs:
        print("No successful runs to blind yet.", file=sys.stderr)
        return 1

    by_doc: dict[str, dict[ModelRef, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_doc[r["doc"]][_ref(r)].append(r)

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "blinded_summaries.md"
    key_path = out_dir / "key.json"

    # Labels are unique across the whole file (doc sections share one
    # A…Z, AA…ZZ sequence) so the key file is a flat label → (model, doc) map.
    key: dict[str, dict] = {}
    label_iter = ("".join(t) for n in (1, 2)
                  for t in itertools.product(string.ascii_uppercase, repeat=n))
    with md_path.open("w") as f:
        f.write(f"# Blinded summaries — {len(by_doc)} doc(s)\n\n")
        f.write("_One run per model per doc (picked randomly from OK runs)._\n\n")
        for doc in sorted(by_doc):
            f.write(f"# Document: {doc}\n\n")
            # Pick one run per model (random if multiple); shuffle so label
            # order doesn't leak the alphabetic model order.
            chosen = [(ref, rng.choice(by_doc[doc][ref]))
                      for ref in sorted(by_doc[doc], key=str)]
            rng.shuffle(chosen)
            for ref, run in chosen:
                label = next(label_iter)
                key[label] = {"model": str(ref), "doc": doc}
                s = run["summary"]
                f.write(f"## {label}\n\n")
                f.write(f"_title {len(s['title'])} chars · description "
                        f"{len(s['description'].split())} words · text "
                        f"{len(s['text'].split())} words_\n\n")
                f.write(f"**title**: {s['title']}\n\n")
                f.write(f"**description**: {s['description']}\n\n")
                f.write(f"**text**:\n\n{s['text']}\n\n---\n\n")

    with key_path.open("w") as f:
        json.dump(key, f, indent=2)

    print(f"Wrote {md_path}")
    print(f"Wrote {key_path}  (keep this aside until judging is done)")
    print(f"\nLabels assigned: {' '.join(key)}")
    return 0


def errors(root: BenchmarkRoot, models: list[ModelRef] | None = None,
           documents: list[str] | None = None) -> int:
    root.require()
    fails = [r for r in load_runs(root, models, documents) if not r.get("ok")]
    if not fails:
        print("No errors.")
        return 0
    print(f"=== Run failures ({len(fails)}) ===")
    for r in fails:
        print(f"  {_ref(r)} · {r['doc']} (run {r.get('run_index')}): "
              f"{r.get('error', '?')}")
        if r.get("raw_output"):
            print(f"    raw: {r['raw_output'][:200]}...")
    return 0
