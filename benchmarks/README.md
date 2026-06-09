# Summarizer benchmark

Pick the best **local** Ollama model for Bartleby's document summarizer:
the fastest one that never breaks structured output and doesn't sacrifice
accuracy. This is a re-runnable selection tool, not a one-off — re-run it
whenever your installed models change.

Why local-only: ingestion runs parse → caption → summarize as three
sequential phases behind barriers, so the summarizer never competes with the
vision model for VRAM. That frees us to choose the summarization model purely
on its own merits. (The accuracy judge is the one cloud component — see below.)

## What it measures

For each candidate model, over a **real Bartleby corpus** (not one document):

1. **Speed** — load-corrected throughput. `tok/s` from Ollama's `eval_duration`,
   plus median inference seconds (`wall_seconds − load_duration`) per doc.
2. **Structured-output reliability** — share of runs that parse into
   `DocumentSummary`. This is a **hard gate**, not a soft metric: a fast model
   that fails the schema is unusable. Schema-failers are dropped from the
   leaderboard regardless of speed.
3. **Accuracy** — a cloud judge scores each summary against its source on
   faithfulness, coverage, conciseness, and constraint-compliance (1–5 each).

## Production parity

The harness imports Bartleby's actual `build_summary_messages` +
`DocumentSummary`, and defaults to the production settings ingest uses:

- **temperature 0.0** (was 1.0 in the old prototype — lower temp also curbs the
  degeneration the temp-1.0 runs showed).
- **`max_summarize_tokens` = 50,000** (matches `DEFAULT_MAX_SUMMARIZE_TOKENS`),
  truncating with the same `tiktoken cl100k_base` encoder as production.

Keeping those imports is what makes the benchmark meaningful: it exercises the
same code path `bartleby ingest` takes.

## Methodology

- **Corpus sampling.** `summarize_local.py` samples documents from a project DB,
  stratified into short/medium/long tertiles by `documents.token_count`, drawing
  evenly across bands and deliberately forcing in at least one doc that exceeds
  `max_tokens` so truncation behavior is exercised. Pin exact docs with
  `--document-ids`.
- **Randomized order.** The whole (model × doc × run) matrix is shuffled into one
  order so thermal throttling and weight-eviction can't favor any model or doc.
  Pass `--seed` to make the plan reproducible.
- **Model discovery.** With no `--models`, the set is auto-discovered from
  `ollama list` minus a skip-list of non-text models (vision `*-vl*`, embeddings
  `nomic-embed*`, image-gen `x/*`, and coder variants). So the set tracks
  whatever you actually have installed. `--include-coder` keeps coders;
  `--models a b c` pins an explicit set.
- **Blind, automated judge.** `judge.py` sends each summary to a cloud model
  (default `gpt-5.5`) with **only** the source text and the summary — never the
  model name or a label — and scores it on the rubric. Independent per-summary
  scoring means neither identity nor position can leak. One run per (model, doc)
  is judged by default; `--all-runs` judges every run.
- **Human spot-check.** Before trusting the leaderboard, validate the judge:
  `examine.py blind` writes blinded summaries + a key file for a doc so you can
  hand-rank a random sample and confirm the judge tracks reality.

## Running it

```sh
# 1. Benchmark (hammers local Ollama — can take hours for a full model set).
uv run python benchmarks/summarize_local.py \
    --db ~/.bartleby/projects/centralhudson-redux/bartleby.db \
    --sample 12 --runs 3 --seed 1 \
    --out benchmarks/results.jsonl

# 2. Inspect raw speed + schema-validity, and any failures.
uv run python benchmarks/examine.py timings benchmarks/results.jsonl
uv run python benchmarks/examine.py errors  benchmarks/results.jsonl

# 3. Judge accuracy (cloud; needs OPENAI_API_KEY in env or benchmarks/.env).
uv run python benchmarks/judge.py benchmarks/results.jsonl \
    --out benchmarks/judged.jsonl

# 4. Final ranked leaderboard.
uv run python benchmarks/examine.py leaderboard benchmarks/results.jsonl \
    --judged benchmarks/judged.jsonl > benchmarks/leaderboard.md

# Optional: blinded summaries for a human spot-check of the judge.
uv run python benchmarks/examine.py blind benchmarks/results.jsonl \
    --document-id 45 --out benchmarks/blind/
```

## Reading the leaderboard

- **Schema-valid %** is the gate. Anything below `--min-schema` (default 100%)
  is listed under *Disqualified* and excluded from the ranking — a model that
  ever breaks structured output can't run the ingest pipeline.
- Among survivors, **★ marks the speed/quality Pareto frontier**: no other
  survivor is both faster *and* higher-quality. The frontier is the real
  shortlist; pick the point on it that fits your speed/quality trade-off.
- Mean quality is the judge's rubric average (out of 5). Treat it as a guide
  validated by the spot-check, not ground truth.

## What's tracked vs. local

Tracked (in version control): `summarize_local.py`, `examine.py`, `judge.py`,
this README.

Git-ignored (local only): `results*.jsonl`, `judged*.jsonl`, `blind*/`, the
legacy `*-judged.csv`, `.env`. The earlier cloud-summarizer variants
(`summarize_openai.py`, `summarize_omlx_vs_ollama.py`) are intentionally kept
out of VC — cloud-cost comparison is out of scope for local summarizer
selection. They may resurface as their own benchmark.
