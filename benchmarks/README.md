# Summarizer benchmark

Pick the best **local** Ollama model for Bartleby's document summarizer: the
fastest one that never breaks structured output and doesn't sacrifice
accuracy. This is a re-runnable selection tool, not a one-off — evidence
accumulates in append-only stores, so re-run it whenever your installed
models change or whenever the machine is idle.

Why local-first: ingestion runs parse → caption → summarize as three
sequential phases behind barriers, so the summarizer never competes with the
vision model for VRAM. That frees us to choose the summarization model purely
on its own merits. Cloud models appear only as **reference rows** — quality
yardsticks that can't win the selection — and the accuracy judge is the one
load-bearing cloud component.

## The commands

Everything runs through the `bartleby benchmark` command group from the repo
root (or pass `--benchmarks-dir`):

```sh
# 1. Append summarize runs: every model in models.yaml × every doc in
#    corpus.yaml, shuffled. Each invocation adds --runs (default 1) records
#    per cell — run it again later and the evidence grows.
bartleby benchmark summarize
bartleby benchmark summarize --models ollama/gemma4:e2b,openai/gpt-5-nano \
    --documents rci-reply-0116 --runs 3

# 2. Top up blind judge scores (cloud; needs OPENAI_API_KEY in env or
#    benchmarks/.env). Declarative and idempotent: each distinct summary is
#    brought up to --passes (default 3) OK judgments; re-running the same
#    value is a no-op. Want more judgments later? Raise the number —
#    --passes 8 adds 5 to everything that has 3.
bartleby benchmark judge
bartleby benchmark judge --model openai/gpt-5.5 --passes 5

# 3. The ranked report. Filters compose; --output also writes CSV.
bartleby benchmark leaderboard
bartleby benchmark leaderboard --since 2026-06-01 --until 2026-06-09 \
    --models ollama/gemma4:e2b --documents opposition-0098 \
    --judges openai/gpt-5.5 --output leaderboard.csv

# Blinded summaries + key file for a human spot-check of the judge.
bartleby benchmark blind --seed 7

# Failed runs, with raw-output previews (how a corrupt model blob gets
# distinguished from a schema failure).
bartleby benchmark errors
```

Model references are always `<provider>/<model>` — YAML configs and command
line alike. The first slash splits, so Ollama names keep their colons:
`ollama/gemma4:e2b`.

## The stores

```
benchmarks/
  corpus.yaml    models.yaml    judges.yaml     # tracked configuration
  corpus/                                       # tracked PDFs
  sources/<doc-id>.txt                          # ignored: extracted-text cache
  results/<provider>_<model>_<doc-id>.jsonl     # ignored: run records, append-only
  judgements/<provider>_<model>_<doc-id>_<judge-provider>_<judge-model>.jsonl
```

Every record carries its own identity (`provider`, `model`, `doc`, the judge
pair) plus provenance: `source_sha` (hash of the exact text the model saw),
`prompt_sha` (hash of the summarize prompt + `DocumentSummary` schema),
`temperature`, `max_tokens`, and `bartleby_version`. Filenames are write-side
organization only — readers glob and filter on record contents.

Nothing is ever rewritten: summarize appends runs with per-cell monotonic
`run_index`es, judge appends passes. The leaderboard's `--since`/`--until`
window is how you cut a report from a particular era, and it **warns** when a
cell's windowed records mix `source_sha`, `prompt_sha`, `temperature`, or
`max_tokens` regimes instead of silently averaging across them. Deleting `sources/` forces re-extraction —
but old runs then carry a stale `source_sha`, and the judge **refuses** to
score a summary against text the model never saw.

## The benchmark corpus

`corpus.yaml` maps doc-ids to the committed PDFs:

- `psc-order-0109` — a 6-page NY PSC order from the Central Hudson rate case
  (~2,200 tokens). Formulaic, templated institutional prose: petitioner's
  alleged errors → the Commission's discussion → the denial.
- `rci-reply-0116` — a 4-page intervenor reply statement (~2,300 tokens).
  Size-matched to the order but a different register: flowing advocacy prose,
  so models can't lean on the order genre's boilerplate structure.
- `opposition-0098` — an 11-page argumentative brief (~8,400 tokens). The
  long document: at this length a schema-constrained summary must actually
  *select*, which makes coverage a live, discriminating axis instead of a
  constant.

All three were chosen deliberately: **image-free** (every page
text-extractable, so the pdfplumber → chunk path reproduces the *whole*
production summary input without the caption pipeline), **coherent prose**
with concrete dates, parties, and case numbers (faithfulness and coverage are
genuinely testable), and **under `max_summarize_tokens`** (no truncation to
muddy the comparison). More docs — including image-bearing ones — and a
Docling-extraction variant are tracked in #242.

## Production parity

`summarize` assembles each model input **exactly the way ingest does** for an
image-free document: extract with the **pdfplumber backend** at production
thresholds, chunk each text page with Bartleby's `chunk_text`, join the
document chunks in page/chunk order — the same result `writer.summary_input`
produces — then truncate with the same `tiktoken cl100k_base` encoder to the
same `DEFAULT_MAX_SUMMARIZE_TOKENS`. Each call uses Bartleby's actual
`build_summary_messages` + `DocumentSummary` schema at **temperature 0.0**
(production's setting; cloud reference rows run at provider defaults —
gpt-5.x rejects pinned temperatures).

## What it measures

1. **Speed** — load-corrected throughput for local models: `tok/s` from
   Ollama's `eval_duration`, inference seconds as `wall − load_duration`,
   reported as the mean of per-doc medians (inference cost tracks doc
   length). Cloud rows carry a `†`: their wall-clock measures a datacenter
   plus the network, so it's recorded but never compared.
2. **Structured-output reliability** — share of runs that parse into
   `DocumentSummary`. This is a **hard gate**, not a soft metric: a model
   that ever breaks the schema is unusable and lands under *Disqualified*
   regardless of speed.
3. **Accuracy** — a cloud judge scores each summary against its source on
   faithfulness, coverage, conciseness, and constraint-compliance (1–5 each).

## Methodology

- **Randomized order.** Each summarize invocation shuffles its (model × doc ×
  run) matrix so thermal throttling and weight-eviction can't favor any
  model. `--seed` makes a plan reproducible.
- **Curated lists, no filtering.** `models.yaml` and `judges.yaml` are the
  source of truth; nothing is auto-discovered or dropped. Listed-but-not-
  installed Ollama models are warned about up front and their runs record an
  error.
- **Blind, automated judge.** The judge sees **only** the cached source text
  and the candidate summary — never the model name — and scores the rubric.
- **Judge noise is measured, not ignored.** Runs dedupe into distinct
  summaries (at temp 0.0 repeats are usually byte-identical), each judged
  `--passes` times. A 1–5 integer rubric draws different scores for the same
  text on different calls — roughly ±1 per axis — so single-pass scores carry
  noise the size of the gaps between good models. Passes average; distinct
  summaries then weight by how many runs produced each; docs weight equally.
- **Per-judge stores.** Scores from different judges never mix silently; add
  an off-family judge to judges.yaml to control for same-family preference
  bias and compare with `--judges`.
- **Human spot-check.** Before trusting the leaderboard, validate the judge:
  `bartleby benchmark blind` writes blinded summaries + a key file so you can
  hand-rank them and confirm the judge tracks reality.

## Reading the leaderboard

- **Schema-valid %** is the gate; failers sit under *Disqualified*. (A model
  that never produced an OK run — e.g. a corrupt blob that won't load — lands
  there too; `bartleby benchmark errors` tells those apart.)
- Among judged **local** survivors, **★ marks the speed/quality Pareto
  frontier**: no other local survivor is both faster *and* higher-quality.
  The frontier is the real shortlist. `†` rows are cloud reference anchors —
  quality context, never candidates.
- The **per-doc quality table** shows each cell's score with its evidence
  (`4.58 (3r/9p)` = 3 OK runs, 9 judge passes) so a thin cell can't
  masquerade as a solid one, and a long-doc coverage gap can't hide inside
  the overall mean.
- Gaps of a couple tenths between models are within judge noise even after
  pass-averaging; treat near-ties as ties and pick by speed.
