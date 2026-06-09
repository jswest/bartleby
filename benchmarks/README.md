# Summarizer benchmark

Pick the best **local** Ollama model for Bartleby's document summarizer:
the fastest one that never breaks structured output and doesn't sacrifice
accuracy. This is a re-runnable selection tool, not a one-off — re-run it
whenever your installed models change.

Why local-only: ingestion runs parse → caption → summarize as three
sequential phases behind barriers, so the summarizer never competes with the
vision model for VRAM. That frees us to choose the summarization model purely
on its own merits. (The accuracy judge is the one cloud component — see below.)

## The benchmark document

This first pass benchmarks **one committed PDF**,
`corpus/0109_Order_Denying_Request_for_Rehearing_and_Reconsideration.pdf` — a
6-page New York PSC order from the Central Hudson rate case (~2,200 tokens).
It's checked in so the benchmark is reproducible by anyone, not dependent on a
local project DB.

It was chosen deliberately:

- **Image-free.** Every page is text-extractable (no embedded images, no sparse
  pages). This matters: with no image chunks, the pdfplumber → chunk path
  reproduces the *whole* production summary input, so the benchmark is faithful
  without dragging in the caption pipeline.
- **Coherent prose.** A self-contained legal narrative (petitioner's alleged
  errors → the Commission's discussion → the denial), with concrete dates,
  parties, and case numbers — so faithfulness and coverage are genuinely
  testable, not measuring table-extraction noise.
- **Medium length.** Well under `max_summarize_tokens`, so there's no truncation
  to muddy the comparison.

Single-doc is a starting point. More docs (including image-bearing ones) and a
Docling-extraction variant are tracked as follow-ups in #242.

## Production parity

`summarize_local.py` assembles the model input **exactly the way ingest does**
for an image-free document: it extracts the PDF with the **pdfplumber backend**
at the production thresholds, chunks each text page with Bartleby's `chunk_text`,
and joins the document chunks in page/chunk order — the same result
`writer.summary_input` produces. (Verified byte-identical to the chunks ingest
stored for this doc.) It then summarizes that input with Bartleby's actual
`build_summary_messages` + `DocumentSummary` schema, at the production settings:

- **temperature 0.0** (a lower temp also curbs the degeneration the old temp-1.0
  prototype showed).
- **`max_summarize_tokens` = 50,000** (matches `DEFAULT_MAX_SUMMARIZE_TOKENS`),
  truncating with the same `tiktoken cl100k_base` encoder as production.

pdfplumber is the only extraction backend here; Docling is deferred.

## What it measures

For each candidate model, over N runs:

1. **Speed** — load-corrected throughput. `tok/s` from Ollama's `eval_duration`,
   plus median inference seconds (`wall_seconds − load_duration`).
2. **Structured-output reliability** — share of runs that parse into
   `DocumentSummary`. This is a **hard gate**, not a soft metric: a fast model
   that fails the schema is unusable. Schema-failers are dropped from the
   leaderboard regardless of speed.
3. **Accuracy** — a cloud judge scores each summary against the source on
   faithfulness, coverage, conciseness, and constraint-compliance (1–5 each).

## Methodology

- **Randomized order.** The whole (model × run) matrix is shuffled into one
  order so thermal throttling and weight-eviction can't favor any model. Pass
  `--seed` to make the plan reproducible.
- **Model list (`models.yaml`).** The set is **exactly** the list under
  `models:` in `benchmarks/models.yaml` — committed to VC, run in order, with
  **no filtering**: nothing is auto-discovered and no model is dropped for being
  a vision/coder/embedding name. Curating it is your job — add or remove lines,
  `ollama pull` what you don't have (or open a PR). Listed-but-not-installed
  models are warned about up front and their runs simply record an error.
  `--models a b c` overrides the file for an ad-hoc run.
- **Blind, automated judge.** `judge.py` sends each summary to a cloud model
  (default `gpt-5.5`) with **only** the source text and the summary — never the
  model name — and scores it on the rubric. The source is the exact (truncated)
  text the models saw, carried in the results config record. One run per model
  is judged by default; `--all-runs` judges every run.
- **Human spot-check.** Before trusting the leaderboard, validate the judge:
  `examine.py blind` writes blinded summaries + a key file so you can hand-rank
  them and confirm the judge tracks reality.

## Running it

Edit `models.yaml` to the set you want, then:

```sh
# 1. Benchmark the committed doc (runs every model in models.yaml; hammers
#    local Ollama — minutes per model).
uv run python benchmarks/summarize_local.py --runs 3 --seed 1 \
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
    --out benchmarks/blind/
```

`--pdf` points the benchmark at a different file if you want to try one ad hoc.

## Live progress

On a terminal, `summarize_local.py` shows a live `rich` view: an overall
matrix progress bar (calls done, elapsed, ETA), a streaming line for the model
currently generating (live token estimate + tok/s + elapsed, with a pulse bar —
generation length isn't known ahead of time, so the bar signals activity, not a
percentage), and a per-model dashboard table (status, runs done, tok/s, schema
gate) that fills in as runs land. The live token count is a tiktoken estimate of
the streamed text; the exact `eval_count`/`tok/s` recorded in the JSONL come
from Ollama's final-chunk metadata. Off a terminal (piped or redirected) it
degrades to one plain `[k/N] model (run i)` line per call, then a `✓/✗` on
completion — no escape codes in your logs.

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
`models.yaml`, this README, and the one benchmark PDF under `corpus/`.

Git-ignored (local only): `results*.jsonl`, `judged*.jsonl`, `blind*/`, the
legacy `*-judged.csv`, `.env`. The earlier cloud-summarizer variants
(`summarize_openai.py`, `summarize_omlx_vs_ollama.py`) are intentionally kept
out of VC — cloud-cost comparison is out of scope for local summarizer
selection. They may resurface as their own benchmark.
