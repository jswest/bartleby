# Bartleby architecture

Bartleby is two surfaces over one SQLite database:

1. **The CLI (`bartleby ...`)** — ingestion (`scribe`), config (`ready`), projects, sessions, embedding, audit logs.
2. **The skill (`skill/`)** — six Python scripts the agent calls: `list_documents`, `search`, `read_chunks`, `read_document`, `save_summary`, `save_finding`.

The database is the contract between them. The CLI writes; the skill reads and writes findings back. Both import from one Python package (`bartleby/`).

For the on-disk shape, read the code: `bartleby/db/schema.py` for the schema, `bartleby/commands/*` for CLI subcommands, `bartleby/skill_scripts/*` for the scripts. Each script's module docstring documents its JSON response contract, and `bartleby skill <name> --help` prints that docstring alongside the flags (the docstring is the parser's `description`), so the agent can introspect both arguments and return shape without running the script.

## Backwards compatibility

**Default position: we don't care about it.** No migration code, no compat shims, no feature-flagged old code paths. Bump `SCHEMA_VERSION`, change the code, tell users to re-ingest. The cost of preserving compat is invariably higher than the cost of re-ingest for a tool at this scale.

**The one allowed relaxation: additive-only schema upgrades.** A schema bump may ship with an entry in the upgrade chain (`bartleby/db/upgrades.py`) if — and only if — the change is purely additive: new tables, new indexes, new nullable columns. No row transformations, no column renames, no semantic shifts in existing data. Users run `bartleby project upgrade <name>` explicitly to apply the chain; the strict version check in `open_db` rejects mismatched DBs otherwise. Non-additive bumps still mean re-ingest (the chain simply has no entry for that step, and `project upgrade` refuses).

The discipline: every new bump is either additive-with-an-upgrade-function or non-additive-with-re-ingest. The codebase never branches on schema version; it always pins to `SCHEMA_VERSION` exactly. The upgrade path is one-shot at the gate, not an ongoing tax.

## Load-bearing invariants

Things that look local but aren't. Code review should catch violations.

### Polymorphic-chunks discipline

`chunks.source_id` is not a foreign key to any single table — it references `documents`, `summaries`, `findings`, or `images` depending on `source_kind`. SQLite can't enforce this; the `CHECK (source_kind IN (...))` constraint blocks typos but not kind/id mismatches.

**All writes to `chunks` go through `bartleby/db/chunks.py` typed helpers** (`insert_document_chunks`, `insert_summary_chunks`, `insert_finding_chunks`, `insert_image_chunks`, `delete_chunks_for`). The chokepoint exists so the `source_kind` is hardcoded per function. Direct `INSERT INTO chunks` anywhere else is a bug.

### Single-writer drain + per-unit resume

Ingest writes flow through **one `Writer` (`bartleby/ingest/writer.py`) that owns the WAL connection** — the sole path that persists a parse, a caption, or a summary, each in its own transaction. The producer side is pure (no DB); the Writer drains every result. Ingest runs in **three sequential phases behind the Writer**: **parse** fans out across a `spawn` process pool (`bartleby/ingest/pool.py`, #165) — workers only parse + embed, never touch the DB; **caption** is its own concurrent stage (#166) — a thread pool runs each image's OCR + VLM (the network/IO work the GIL releases) off the writer thread, while embedding the caption and writing it stay on the main thread; **summarize** is a separate pass (`_summarize_all`, #167/#188) over every document the DB still lacks a summary for (`Writer.documents_needing_summary`) — lifted off the parse/caption path so the slow per-doc LLM summary (≈59% of wall-clock per the #162 benchmark) can't throttle the earlier phases, and (#188) fans out across its own `summarize_workers` thread pool exactly as caption does: the LLM call on worker threads, every Writer call (including the `summary_input` payload read) on the main thread. All three fan-outs feed the one `Writer`, which persists every unit single-threaded (apsw connections aren't thread-safe), so all concurrency lives upstream of the DB. Parse processes are RAM-bound and auto-sized (`max_workers`, which holds back a couple of cores) and **recycle after `WORKER_MAX_TASKS` documents** so docling/torch RSS can't grow unbounded over a long run (#211/#213); caption and summarize threads are network/IO-bound flat counts (`caption_workers` / `summarize_workers`, default 4 each for a cloud provider but **auto-clamped to 1 for a local Ollama provider**, which serializes — `OLLAMA_NUM_PARALLEL` defaults to 1, #243) — `--timings` pins all three to 1 for a clean sequential baseline. The split is load-bearing for **restartability**: a document's parse (text chunks + *uncaptioned* image rows) commits atomically and is never re-parsed; each image **caption** (keyed by image byte-hash) and the **summary** commit independently, so a caption failure can't roll back the parse. Completeness is derived from DB state — `parsed ∧ every image row has `analysis_json` ∧ (summary row exists ∨ summaries disabled ∨ 0 summarizable chunks)` — and a resume runs only the missing units: the parse/caption phases resume whatever images are still uncaptioned, and the summarize pass independently picks up whatever documents still owe a summary (so a corpus parsed with summaries off is summarized in full on a later run with them on, no re-parse). An image row with `analysis_json IS NULL` is the uncaptioned marker; do not write a non-null analysis at parse time. A deterministically-failing unit is recorded in `failed_ingests` and capped at `MAX_INGEST_ATTEMPTS` (not retried forever), and surfaced at end-of-run and in `bartleby project info` — a skipped unit must never read as a completed one. There is intentionally **no stranded-partial cleanup**: atomic parse means a `documents` row implies a finished parse, so the old `_document_already_ingested` / `_cleanup_partial_document` dance is gone.

Because a resume can re-run units across separate invocations, each persisted unit also carries **provenance**: `Writer.begin_run` records the run's resolved, secret-stripped config in the `ingests` table and stamps the document / summary / chunk rows it produces with that `ingest_run_id` (nullable — NULL on the skill write path and on corpora ingested before this feature). On a re-run the producer warns — never blocks — on any config field that drifted from the last ingest. See [the ingest-provenance decision](./docs/decisions/GH-0171-unit-ingest-provenance-config-drift-warnings-0001.md).

### Image content_type discipline

Image chunks carry one of two `content_type` values: `image_ocr` (verbatim transcription, treat as primary source) or `image_description` (model interpretation of visual content, cite as interpretation). The split is enforced by `bartleby/ingest/images.py:analysis_to_chunk_inputs` and surfaced to agents via SKILL.md. Both `chunks` rows for an image point at the same `image_id` via `chunks.source_id`; the document anchor lives in `document_images` (one join row per occurrence).

### Summarizer structured-output contract

The summarizer is structured-output only across all three providers (Anthropic / OpenAI / Ollama). Even though we control the prompt, JSON enforcement keeps open-source models from drifting into "Here is your summary:" preambles, thinking tags, or stray markdown fences. The schema is rendered from a Pydantic model (`DocumentSummary`); all three providers consume `model_json_schema()`, so adding fields means one place to change.

Validation failures raise. Don't insert malformed summaries silently.

### Memory-off enforcement

A session with `memory_enabled = 0` must not see *other sessions'* findings — enforced at the **script level**, not the prompt level. Two enforcement shapes: `skill_scripts/search.py` silently excludes **all** findings from results regardless of flags (ranked retrieval is the contamination vector — don't soften this); the direct-read commands `read_finding` / `list_findings` instead narrow to the session's **own** findings (a run can read back what it wrote, never another session's), and `read_chunks` applies the same wall to finding-kind chunks reached by id — foreign ones fall into its `missing` list (`--chunks`) or raise `MEMORY_OFF` (`--around-chunk`), since chunk ids would otherwise be a back door to other sessions' finding bodies. `edit_finding` gates the same way — it echoes the stored body in its response, so an ungated edit on a foreign finding is a read-by-write bypass (GH-0272). `delete_finding` gates too (its response echoes the deleted title, and it destroys the finding outright), and `merge_findings` gates across every finding it touches — the `--into` target and all `--from` sources — since it consumes and erases them (GH-0275, superseding GH-0056's originally-ungated stance for these two). Only `save_finding` stays open — it authors the session's *own* finding and discloses nothing foreign — so a memory-off run can still produce findings for later comparison. The line: a command gates iff it would disclose, mutate, or destroy a finding the caller didn't author.

### Truncation note

When a document exceeds `max_summarize_tokens`, the summary's `text` field gets a deterministic note appended **in code**, not via the prompt. The caveat is guaranteed, not modeled.

## Conventions

- Skill scripts print one JSON object to stdout, exit non-zero on error with `{"error", "code"}`. Prose/progress goes to stderr only.
- Embedding model: `BAAI/bge-base-en-v1.5` (768 dims, 512 token max). FTS5 tokenizer: `unicode61 remove_diacritics 2`.
- LLM provider defaults: anthropic `claude-haiku-4-5`, openai `gpt-5-mini`, ollama `qwen3-vl:30b`.
- VLM provider defaults: anthropic `claude-haiku-4-5`, openai `gpt-5-mini`, ollama `qwen3-vl:30b`.
- PDF converter (config `pdf_converter`, CLI `--pdf-converter`): `pdfplumber` (default — fast text + page-render image extraction) and `docling` (opt-in — better structural extraction at higher cost).
- HTML converter (config `html_converter`, CLI `--html-converter`): `docling` (default) and `sec2md` (opt-in — routes iXBRL EDGAR filings to sec2md by sniff, non-iXBRL HTML falls back to docling). MD always goes through docling. If you have an HTML/MD corpus, `docling` must be installed; if you set `html_converter=sec2md`, the `sec2md` extra must also be installed.
- Dependency management: `uv` (not pip/venv). Run with `uv run python`.
- **Skill flag naming (issue #111)**: id-taking flags name the *noun*; the value is its id — **no `-id` suffix** (`--document`, `--finding`, `--from`, `--into`, `--chunks`, `--around-chunk`). The internal argparse `dest` may still be `finding_id` etc.; only the user-facing flag drops the suffix. **Arity is carried by singular/plural**: a singular flag is one id (`--document <id>`, `--finding <id>`), a plural flag is a comma-list (`--documents <id,id>`, `--chunks <id,id>`). **Scope vs target are deliberately distinct** even when both are plural id-lists: `--in-documents` is a *scope filter* (narrows `search`/`scan`/`describe_corpus`/`list_documents` to that set) while `--documents` is a *target set* (the docs `assign_tag`/`unassign_tag` act on). **Merge verbs agree**: destination is `--into`, source is `--from`, on both `merge_findings` and `merge_tags`.

## Deferred (potential v2)

Things we said no to and may revisit:

- Map-reduce summarization. Currently single-shot only.
- Cross-session memory beyond "search past findings" — no automatic injection, no summarization of past sessions.
- Per-project config beyond `~/.bartleby/config.yaml`.
- Porter stemming for FTS5 (`porter unicode61 remove_diacritics 2`). Better recall, requires re-index.
- An MCP server (the skill is plain scripts).
- Differentiating agent-saved summaries from ingest-time summaries (would need a `created_by` column on `summaries`).

## Decision log

Settled judgment calls now live one-per-file under
[`docs/decisions/`](./docs/decisions/) — read
[`docs/decisions/README.md`](./docs/decisions/README.md) for the chronological
index (newest first). This file holds **current state only**; the *why* behind a
past call lives in its decision file.
