#!/usr/bin/env python
"""Automated accuracy judge for benchmarks/summarize_local.py output.

Reads a results JSONL, and for each successful model summary asks a strong
cloud model (OpenAI, default ``gpt-5.5``) to score it against the source
document on a four-axis rubric:

  - faithfulness        — no invented figures, dates, names, or claims
  - coverage            — captures the document's key facts and structure
  - conciseness         — dense and non-redundant, no filler
  - constraint_compliance — honors DocumentSummary's stated limits
                            (title <= 60 chars, description ~20 words / <= 200 chars)

Each axis is scored 1-5 with a one-line rationale. The judge is **blind by
construction**: its prompt contains only the source text and the candidate
summary, never the model name, and each summary is scored independently. The
source is the exact (truncated) text the models saw, carried in the results
config record. Validate the leaderboard with a human spot-check
(``examine.py blind``) before trusting it.

By default one run per model is judged (temp-0.0 outputs are near-identical);
pass --all-runs to judge every recorded run.

Reads OPENAI_API_KEY from the environment or benchmarks/.env.

Example:
  uv run python benchmarks/judge.py benchmarks/results.jsonl --out benchmarks/judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


DEFAULT_JUDGE_MODEL = "gpt-5.5"
RUBRIC_AXES = ("faithfulness", "coverage", "conciseness", "constraint_compliance")


class JudgeScore(BaseModel):
    """Per-summary rubric the judge fills in. Each axis is 1 (worst)-5 (best)."""

    faithfulness: int = Field(ge=1, le=5, description="No invented figures, dates, names, or claims.")
    coverage: int = Field(ge=1, le=5, description="Captures the document's key facts and structure.")
    conciseness: int = Field(ge=1, le=5, description="Dense and non-redundant; no filler.")
    constraint_compliance: int = Field(
        ge=1, le=5,
        description="Honors the schema limits: title <= 60 chars, description ~20 words / <= 200 chars.",
    )
    rationale: str = Field(description="One or two sentences justifying the scores; cite a specific flaw if any.")


# Framing only — the per-axis definitions and the exact constraint thresholds
# live in the JudgeScore field descriptions, which reach the model through the
# response_format schema. Don't restate them here or the two copies will drift.
JUDGE_INSTRUCTIONS = (
    "You are grading a machine-generated summary against its source document. "
    "Score each rubric axis from 1 (worst) to 5 (best) per its definition, then "
    "give a one-line rationale. Be strict and specific. You do not know which "
    "model wrote this summary; judge only what is in front of you."
)


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
                print(f"warning: skipping malformed line in {path}", file=sys.stderr)
    return records


def _config(records: list[dict]) -> dict:
    return next((r for r in records if r.get("kind") == "config"), {})


def _build_prompt(source_text: str, summary: dict) -> list[dict]:
    candidate = (
        f"title: {summary['title']}\n"
        f"description: {summary['description']}\n"
        f"text:\n{summary['text']}"
    )
    return [
        {"role": "system", "content": JUDGE_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"SOURCE DOCUMENT:\n{source_text}\n\n"
                f"CANDIDATE SUMMARY:\n{candidate}"
            ),
        },
    ]


def judge_summary(client: OpenAI, model: str, source_text: str, summary: dict) -> dict:
    wall_start = time.perf_counter()
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=_build_prompt(source_text, summary),
            response_format=JudgeScore,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "wall_seconds": time.perf_counter() - wall_start}
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return {"ok": False, "error": "judge returned no parsed payload",
                "wall_seconds": time.perf_counter() - wall_start}
    scores = parsed.model_dump()
    scores["mean"] = sum(scores[a] for a in RUBRIC_AXES) / len(RUBRIC_AXES)
    return {"ok": True, "scores": scores, "wall_seconds": time.perf_counter() - wall_start}


def select_runs(records: list[dict], all_runs: bool) -> list[dict]:
    """The OK runs to judge: one per model by default, else every run."""
    ok = [r for r in records if r.get("kind") == "run" and r.get("ok")]
    if all_runs:
        return ok
    seen: set[str] = set()
    chosen: list[dict] = []
    for r in ok:
        if r["model"] not in seen:
            seen.add(r["model"])
            chosen.append(r)
    return chosen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", type=Path, help="JSONL from summarize_local.py")
    p.add_argument("--out", type=Path, required=True, help="JSONL output path for judge scores")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Cloud model used to score (default {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--all-runs", action="store_true",
                   help="Judge every recorded run, not just one per model")
    args = p.parse_args(argv)

    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (looked in env and benchmarks/.env)")

    records = load_records(args.results)
    source_text = _config(records).get("source_text")
    if not source_text:
        raise SystemExit("Results config has no source_text — re-run summarize_local.py.")

    to_judge = select_runs(records, args.all_runs)
    if not to_judge:
        raise SystemExit("No successful runs to judge.")

    client = OpenAI()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")

    def write(record: dict):
        record["timestamp"] = time.time()
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

    write({"kind": "config", "results": str(args.results),
           "judge_model": args.judge_model, "all_runs": args.all_runs,
           "count": len(to_judge)})

    print(f"Judging {len(to_judge)} summary(ies) with {args.judge_model}...",
          file=sys.stderr)
    tallies: dict[str, list[float]] = defaultdict(list)
    for idx, run in enumerate(to_judge, 1):
        model = run["model"]
        print(f"  [{idx}/{len(to_judge)}] {model} (run {run.get('run_index')})",
              file=sys.stderr, flush=True)
        result = judge_summary(client, args.judge_model, source_text, run["summary"])
        if result.get("ok"):
            tallies[model].append(result["scores"]["mean"])
        write({"kind": "judgment", "model": model,
               "run_index": run.get("run_index"), **result})

    out_f.close()
    print(f"\nWrote {len(to_judge)} judgment(s) to {args.out}", file=sys.stderr)
    for model in sorted(tallies):
        scores = tallies[model]
        print(f"  {model}: mean {sum(scores) / len(scores):.2f} over {len(scores)} run(s)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
