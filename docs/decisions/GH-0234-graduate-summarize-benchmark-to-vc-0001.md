# The local-summarizer benchmark is graduated into version control, local-only and judge-automated.

`benchmarks/` was wholly git-ignored — a prototype that lived on one laptop. It
is now a tracked, reproducible model-selection tool: `.gitignore` is narrowed
from a blanket `benchmarks/` to line-item ignores for *outputs and secrets only*
(`benchmarks/results*`, `benchmarks/judged*`, `benchmarks/blind*`,
`benchmarks/*-judged.csv`, `benchmarks/.env`), so `summarize_local.py`,
`examine.py`, `judge.py`, `README.md`, **and one committed benchmark PDF under
`corpus/`** are checked in while raw JSONL and blinded dumps stay local.
Rationale: a selection tool we re-run as installed models change should be
reproducible by anyone, not a single-laptop artifact — which means the input
document travels with the tool, not a dependency on someone's local project DB.

The first pass benchmarks **one committed document**: an image-free, medium
(~2,200-token) NY PSC order. Image-free is the load-bearing property — with no
image chunks, the pdfplumber → chunk path reproduces the *whole* production
summary input, so the benchmark is faithful without involving the caption
pipeline. The harness assembles that input **the same way ingest does**:
`summarize_local.py` extracts with the **pdfplumber backend** at the production
thresholds, chunks each text page with Bartleby's `chunk_text`, and joins the
document chunks in page/chunk order — the exact result `writer.summary_input`
produces (verified byte-identical to the chunks ingest stored for this doc). It
runs at **temperature 0.0** and the production **`max_summarize_tokens` =
50,000** (the old prototype ran temp 1.0 / 16k — both stale), and auto-discovers
candidates from `ollama list` minus a skip-list (vision `*-vl*`, embeddings
`nomic-embed*`, image-gen `x/*`, coder variants). It imports Bartleby's actual
`build_summary_messages` + `DocumentSummary` so the call path matches ingest.
pdfplumber is the only extraction backend; Docling is deferred. A single doc,
extra docs (incl. image-bearing), and a Docling variant are explicit follow-ups.

Accuracy judging is automated. `judge.py` sends each summary plus its source to
a cloud model (OpenAI `gpt-5.5`) for a 1–5 rubric (faithfulness, coverage,
conciseness, constraint-compliance), **blind by construction** — the judge sees
only source + summary, never the model name, and scores each independently.
`examine.py blind` is retained for a human spot-check that validates the judge
before the leaderboard is trusted. **Structured-output reliability is a hard
gate, not a soft metric**: `examine.py leaderboard` drops any model below
`--min-schema` (default 100%) into a *Disqualified* section regardless of speed,
then stars the speed/quality Pareto frontier among the survivors.

Two deliberate scope calls. (1) **Local-only**: the earlier cloud-summarizer
variants (`summarize_openai.py`, `summarize_omlx_vs_ollama.py`) are kept *out* of
VC (git-ignored) — cloud-cost comparison is a non-goal of local summarizer
selection and can be its own benchmark later; the OpenAI judge is the sole
sanctioned cloud component. Consequently `examine.py` was stripped of its
OpenAI per-call cost machinery (local models are always $0). (2) **The real
leaderboard run is deferred**: a full run hammers local Ollama for hours and
Ollama is often contended, so this change ships the *tooling* and methodology;
the committed-numbers run lands as a follow-up the maintainer triggers when
Ollama is free (issue #234).
