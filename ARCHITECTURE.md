# Bartleby architecture

Bartleby is two surfaces over one SQLite database:

1. **The CLI (`bartleby ...`)** — ingestion (`scribe`), config (`ready`), projects, sessions, embedding, audit logs.
2. **The skill (`skill/`)** — six Python scripts the agent calls: `list_documents`, `search`, `read_chunks`, `read_document`, `save_summary`, `save_finding`.

The database is the contract between them. The CLI writes; the skill reads and writes findings back. Both import from one Python package (`bartleby/`).

For the on-disk shape, read the code: `bartleby/db/schema.py` for the schema, `bartleby/commands/*` for CLI subcommands, `bartleby/skill_scripts/*` for the scripts. Each script's docstring documents its JSON contract; `bartleby <cmd> --help` documents flags.

## Backwards compatibility

**We don't care about it.** No migration code, no schema upgraders, no compat shims, no feature-flagged old code paths. Bump `SCHEMA_VERSION`, change the code, tell users to re-ingest. The cost of preserving compat is invariably higher than the cost of re-ingest for a tool at this scale.

## Load-bearing invariants

Things that look local but aren't. Code review should catch violations.

### Polymorphic-chunks discipline

`chunks.source_id` is not a foreign key to any single table — it references `documents`, `summaries`, `findings`, or `images` depending on `source_kind`. SQLite can't enforce this; the `CHECK (source_kind IN (...))` constraint blocks typos but not kind/id mismatches.

**All writes to `chunks` go through `bartleby/db/chunks.py` typed helpers** (`insert_document_chunks`, `insert_summary_chunks`, `insert_finding_chunks`, `insert_image_chunks`, `delete_chunks_for`). The chokepoint exists so the `source_kind` is hardcoded per function. Direct `INSERT INTO chunks` anywhere else is a bug.

### Image content_type discipline

Image chunks carry one of two `content_type` values: `image_ocr` (verbatim transcription, treat as primary source) or `image_description` (model interpretation of visual content, cite as interpretation). The split is enforced by `bartleby/ingest/images.py:analysis_to_chunk_inputs` and surfaced to agents via SKILL.md. Both `chunks` rows for an image point at the same `image_id` via `chunks.source_id`; the document anchor lives in `document_images` (one join row per occurrence).

### Summarizer structured-output contract

The summarizer is structured-output only across all three providers (Anthropic / OpenAI / Ollama). Even though we control the prompt, JSON enforcement keeps open-source models from drifting into "Here is your summary:" preambles, thinking tags, or stray markdown fences. The schema is rendered from a Pydantic model (`DocumentSummary`); all three providers consume `model_json_schema()`, so adding fields means one place to change.

Validation failures raise. Don't insert malformed summaries silently.

### Memory-off enforcement

A session with `memory_enabled = 0` must not see findings in search — enforced at the **script level**, not the prompt level. `skill_scripts/search.py` silently excludes findings regardless of flags. Don't soften this.

### Truncation note

When a document exceeds `max_summarize_tokens`, the summary's `text` field gets a deterministic note appended **in code**, not via the prompt. The caveat is guaranteed, not modeled.

## Conventions

- Skill scripts print one JSON object to stdout, exit non-zero on error with `{"error", "code"}`. Prose/progress goes to stderr only.
- Embedding model: `BAAI/bge-base-en-v1.5` (768 dims, 512 token max). FTS5 tokenizer: `unicode61 remove_diacritics 2`.
- LLM provider defaults: anthropic `claude-haiku-4-5`, openai `gpt-5-mini`, ollama `gpt-oss:20b`.
- VLM provider defaults: anthropic `claude-haiku-4-5`, openai `gpt-5-mini`, ollama `qwen2.5-vl:7b`.
- PDF backends: `pdfplumber` (default — fast text + page-render image extraction) and `docling` (opt-in via config or `--backend docling` — better structural extraction at higher cost). HTML/MD always go through Docling; if you have an HTML/MD corpus, `docling` must be installed.
- Dependency management: `uv` (not pip/venv). Run with `uv run python`.

## Deferred (potential v2)

Things we said no to and may revisit:

- Map-reduce summarization. Currently single-shot only.
- Cross-session memory beyond "search past findings" — no automatic injection, no summarization of past sessions.
- Per-project config beyond `~/.bartleby/config.yaml`.
- Porter stemming for FTS5 (`porter unicode61 remove_diacritics 2`). Better recall, requires re-index.
- An MCP server (the skill is plain scripts).
- Differentiating agent-saved summaries from ingest-time summaries (would need a `created_by` column on `summaries`).

## Decision log

Settled judgment calls, kept here so we don't re-derive them.

- **CLI rename**: `bartleby read` → `bartleby scribe`. "Read" belongs to the agent; the CLI scribes.
- **Sequential ingestion**: `ProcessPoolExecutor` removed. Simpler, predictable, easier to debug.
- **Docling is the only converter** for PDF/HTML/MD. Handles image-PDF OCR internally. Playwright, PyMuPDF, direct Tesseract dropped. `.txt` uses a simple character chunker (Docling has no text reader).
- **Reranker dropped**: cross-encoder reranker removed; RRF over FTS+semantic replaces it.
- **`finding_citations` is chunk-level**: findings cite chunk_ids, not document_ids — preserves "show me exactly the passage" precision.
- **Sessions auto-create**: explicit `bartleby session start` is only needed for `--no-memory`. Otherwise the skill auto-creates a session on first call.
- **Query embedding**: skill shells out to `bartleby embed` via list-form `subprocess.run`. No shell escaping. No daemon.
- **`documents.token_count`**: computed via `tiktoken.cl100k_base`. Approximate across providers; acceptable for a `--force` gate.
- **Schema v2 — title/description on summaries and findings**: `summaries.{title, description}` and `findings.description` (all NOT NULL) so `list_documents` and finding browsing aren't filename-only. Summarizer returns all three fields in one structured-output call — we don't pay for the document text three times.
- **Schema v3 — image pipeline**: added `images` (one row per unique image, deduped on `file_hash`) and `document_images` (one row per occurrence, with `page_number` + `image_index_on_page`). `chunks.source_kind` gains `'image'`. New content_type values: `'ocr'` for Tesseract-recovered scanned-page text (stored under `source_kind='document'`); `'image_ocr'` and `'image_description'` for VLM outputs (stored under `source_kind='image'`). Default PDF backend swapped from Docling to pdfplumber for ~10x faster text-PDF ingest.
- **`search --in-documents`**: scopes a search to those documents' chunks and their summaries' chunks. Findings dropped when set (they're not tied to documents).
- **`search` triage signals**: each hit carries `rank` (1-indexed) and `normalized_score` (top hit = 1.0). Raw RRF `score` is tiny by design (~`0.015–0.033`) and only comparable within one query.
- **`read_chunks --chunks <ids>`**: second mode for direct chunk lookup by id, mutually exclusive with `--document`.
- **No running citation tracker**: an agent asked for `cite <chunk_id>` + `save_finding --use-tracked-citations`. Declined — `save_finding` already gives durable storage; SKILL.md nudges agents to write interim findings when context grows long.
