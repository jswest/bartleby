# The summarizer benchmark is a `bartleby benchmark` command group over append-only per-cell stores.

The three standalone scripts (`summarize_local.py`, `judge.py`, `examine.py`)
are deleted — per the no-backcompat agreement, not shimmed — and the benchmark
moves into the installed CLI as `bartleby benchmark
summarize|judge|leaderboard|blind|errors`, with logic in `bartleby/benchmark/`
and real tests (ending the untested-scripts status GH-0234 sanctioned, which
was premised on them *not* being part of the CLI).

**Append-only per-cell stores replace session-shaped monoliths.** A results
file is one (provider, model, doc) cell; a judgements file adds the judge
pair. Summarize *appends* runs (per-cell monotonic `run_index`), judge
*appends* passes — nothing is ever rewritten, so evidence accumulates across
invocations: add a model without re-running eleven others, add three runs
when the machine is idle, compare judges side by side. Filenames are
write-side organization only; every record carries its own identity fields,
so readers never parse names. Reports cut eras with `--since`/`--until`.

**Idempotent judging, dedupe-first.** OK runs dedupe into distinct summaries
by `summary_sha` (temp-0.0 repeats are usually byte-identical — the first
multi-pass run showed the judge giving the *same text* scores ±1 per axis, so
single-pass scores carry noise the size of the gaps between good models).
`judge` tops each distinct summary up to `--passes` OK judgments and is a
no-op when nothing is owed: re-running never duplicates spend. Aggregation:
passes average per summary; summaries weight by OK-run frequency *recomputed
from the results store at read time* (judgment-side counts go stale as runs
accumulate); docs weight equally.

**Provenance is recorded, drift is loud.** Extraction happens once per doc
into `sources/<doc-id>.txt`; every run record carries `source_sha`,
`prompt_sha`, `temperature`, `max_tokens`, `bartleby_version`. The judge
verifies `source_sha` before scoring and refuses to grade a summary against
text the model never saw; the leaderboard warns when a cell's windowed
records mix sha regimes instead of silently averaging across them.

**Cloud models are reference rows, not candidates.** `openai:` entries
(e.g. `openai:gpt-5-nano`) run the same `DocumentSummary` structured output
as a quality yardstick, but at provider-default sampling (gpt-5.x rejects
pinned temperatures), with no local-comparable throughput, marked `†`, and
never on the Pareto frontier — the frontier compares only what runs on this
machine. The local-only *selection* principle of GH-0234 stands; this adds
calibration, not competition. Caveat recorded: the default judge (gpt-5.5)
shares a family with OpenAI reference rows; `judges.yaml` + per-judge stores
exist so an off-family judge can be added to control for self-preference.

**Reference syntax.** Models are `<provider>/<model>` everywhere — YAML and
command line alike, parsed on the first slash so Ollama names keep their
colons (`ollama/gemma4:e2b`). One form, no per-surface translation. Slugs
normalize `:`/`/` → `-` with collision refusal.

Corpus grows to three committed docs (`corpus.yaml`: formulaic PSC order,
size-matched advocacy reply, ~8.4k-token opposition brief) so coverage is a
live axis. (Issue #268; supersedes the single-doc, session-shaped harness of
GH-0234's first pass. Image-bearing docs and Docling remain #242.)
