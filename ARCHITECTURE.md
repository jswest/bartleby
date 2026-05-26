# Bartleby architecture

Bartleby is two surfaces over one SQLite database:

1. **The CLI (`bartleby ...`)** — ingestion (`scribe`), config (`ready`), projects, sessions, embedding, audit logs.
2. **The skill (`skill/`)** — six Python scripts the agent calls: `list_documents`, `search`, `read_chunks`, `read_document`, `save_summary`, `save_finding`.

The database is the contract between them. The CLI writes; the skill reads and writes findings back. Both import from one Python package (`bartleby/`).

For the on-disk shape, read the code: `bartleby/db/schema.py` for the schema, `bartleby/commands/*` for CLI subcommands, `bartleby/skill_scripts/*` for the scripts. Each script's docstring documents its JSON contract; `bartleby <cmd> --help` documents flags.

## Backwards compatibility

**Default position: we don't care about it.** No migration code, no compat shims, no feature-flagged old code paths. Bump `SCHEMA_VERSION`, change the code, tell users to re-ingest. The cost of preserving compat is invariably higher than the cost of re-ingest for a tool at this scale.

**The one allowed relaxation: additive-only schema upgrades.** A schema bump may ship with an entry in the upgrade chain (`bartleby/db/upgrades.py`) if — and only if — the change is purely additive: new tables, new indexes, new nullable columns. No row transformations, no column renames, no semantic shifts in existing data. Users run `bartleby project upgrade <name>` explicitly to apply the chain; the strict version check in `open_db` rejects mismatched DBs otherwise. Non-additive bumps still mean re-ingest (the chain simply has no entry for that step, and `project upgrade` refuses).

The discipline: every new bump is either additive-with-an-upgrade-function or non-additive-with-re-ingest. The codebase never branches on schema version; it always pins to `SCHEMA_VERSION` exactly. The upgrade path is one-shot at the gate, not an ongoing tax.

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
- PDF converter (config `pdf_converter`, CLI `--pdf-converter`): `pdfplumber` (default — fast text + page-render image extraction) and `docling` (opt-in — better structural extraction at higher cost).
- HTML converter (config `html_converter`, CLI `--html-converter`): `docling` (default) and `sec2md` (opt-in — routes iXBRL EDGAR filings to sec2md by sniff, non-iXBRL HTML falls back to docling). MD always goes through docling. If you have an HTML/MD corpus, `docling` must be installed; if you set `html_converter=sec2md`, the `sec2md` extra must also be installed.
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
- **Schema v4 — `chunks.page_number`**: first-class `INTEGER` column on `chunks`. Populated for pdfplumber-extracted document chunks (the previous `section_heading = "page N"` hack is gone — section_heading is now `NULL` for pdfplumber). NULL for docling chunks (docling doesn't expose per-chunk pages in our wiring), summaries, findings, and image chunks (image page lives on `document_images`). `search` and `read_chunks` surface it directly; `save_finding` returns it on `citations[]`.
- **Tesseract owns image transcription, VLM owns description.** Each embedded image is dispositioned by Tesseract first: if it yields ≥ 300 chars at avg confidence ≥ 60, classification is `kind='text'` and the VLM is never called — Tesseract output becomes the single `image_ocr` chunk. Otherwise classification is `kind='scene'` and the VLM produces a bounded description (~200 words, never asked to transcribe) which becomes the single `image_description` chunk. One chunk per image, not two. The split exists because asking a small local VLM (gemma3, etc.) to verbatim-transcribe a text-heavy slide under JSON-grammar constraint causes per-image latency in the tens of minutes; Tesseract handles it in a second. Sparse-page renders that already failed page-level OCR pass their `OcrResult` down so the image pipeline doesn't re-OCR the same bytes. Providers' `analyze_image` returns a `VlmDescription` (description + notes only); the orchestrator merges it with the Tesseract result into the persisted `ImageAnalysis`.
- **`search --in-documents`**: scopes a search to those documents' chunks and their summaries' chunks. Findings dropped when set (they're not tied to documents).
- **`search` triage signals**: each hit carries `rank` (1-indexed) and `normalized_score` (top hit = 1.0). Raw RRF `score` is tiny by design (~`0.015–0.033`) and only comparable within one query.
- **`read_chunks --chunks <ids>`**: second mode for direct chunk lookup by id, mutually exclusive with `--document`.
- **No running citation tracker**: an agent asked for `cite <chunk_id>` + `save_finding --use-tracked-citations`. Declined — `save_finding` already gives durable storage; SKILL.md nudges agents to write interim findings when context grows long.
- **Additive-only schema upgrades allowed (relaxation of "no backwards compat")**: schema bumps may ship with an entry in `bartleby/db/upgrades.py` if the change is purely additive (new tables, new indexes, new nullable columns). Users invoke `bartleby project upgrade <name>` explicitly; the strict version check in `open_db` is unchanged and still rejects mismatched DBs without that step. Non-additive bumps still force re-ingest (no chain entry). The codebase never branches on schema version — `SCHEMA_VERSION` stays pinned. Rationale: existing users with multi-hour ingests get a graceful path for genuinely safe changes (e.g. `summaries.authored_date` from #15, `tags` table from #16) without us paying the ongoing migration-code tax.
- **Summarizer input includes image chunks**: ingest-time summarization feeds the LLM document chunks *and* image chunks (the VLM's `image_description` or Tesseract's `image_ocr`) interleaved by `(page_number, chunk_index)` rather than the raw extracted body alone. Image-heavy docs (slide decks, figure-laden papers, OCR-fallback pages) get the VLM's already-paid-for analysis folded into the summary input. Implemented in `_build_summary_input` in `bartleby/commands/scribe.py`; standalone-image and text-only docs short-circuit to the original `full_text`. Decorative-image bloat is absorbed by the `max_summarize_tokens` cap (default 50k).
- **Schema v5 — `summaries.authored_date`**: nullable `TEXT` column populated by the summarizer (additional field on the `DocumentSummary` Pydantic model). One LLM call still produces title/description/text/authored_date — no extra inference cost. Strict `YYYY-MM-DD` validation lives in `bartleby.ingest.summarize.normalize_authored_date`; malformed inputs (`"Q3 2024"`, `"2024-13-01"`, partial dates) silently become NULL rather than failing the ingest. Surfaced in `list_documents` (brief mode) and accepted by `save_summary --authored-date`. Shipped with `_upgrade_v4_to_v5` in `bartleby/db/upgrades.py` so existing corpora can opt in via `bartleby project upgrade <name>`; the column stays NULL until the user re-summarizes.
- **Schema v6 — `tags` + `document_tags`**: a controlled vocabulary the user curates (human-named, human-described) with LLM-assisted assignment against document summaries. All operations live in the skill — there is no parallel CLI surface. The agent drives the workflow; humans direct through conversation. Six new skill scripts (`read_tags`, `add_tag`, `delete_tag`, `rename_tag`, `merge_tags`, `tag`) plus `--tag` filters on `list_documents` and `search`. The classifier reuses the configured summarizer provider/model (no new config knob) via a generic `Provider.classify(prompt, schema)` method. `add_tag` does its own embedding-similarity check (BGE, cosine threshold ~0.85) plus normalized-name exact-match against existing tags, and returns `{status: "conflict", similar_to: ...}` on hit instead of creating — surfaced to the human, not silently overridden. `tag --all` reuses the summarizer model once per document; SKILL.md requires human confirmation before invoking it. Shipped with `_upgrade_v5_to_v6`.
- **Converter axis split + sec2md (issue #14)**: the single `backend` config key and `--backend` CLI flag are split into orthogonal axes — `pdf_converter` (`pdfplumber` / `docling`) and `html_converter` (`docling` / `sec2md`). `sec2md` is a third converter slotted into the HTML branch for iXBRL EDGAR filings; it activates only when `html_converter=sec2md` AND a per-file iXBRL sniff (`xmlns:ix="http://www.xbrl.org/.../inlineXBRL"` in the first ~4KB) hits. Non-iXBRL HTML falls through to docling. Hard rename per the no-backwards-compat rule — users re-run `bartleby ready`. No schema change: sec2md chunks land under `source_kind='document'` with new `content_type` values `'sec_text'` / `'sec_table'` (split on `Chunk.has_table`). `section_heading` maps to `Chunk.header` directly (often `None` on 8-Ks, populated when sec2md detects section context). XBRL tags, element IDs, and filing-level metadata (CIK / ticker / form type / filing date) are deliberately deferred — they want a schema bump and belong to #10. SGML `.txt` full-submission wrappers are not supported by sec2md and are out of scope. New optional dep: `bartleby[sec2md]`, pinned `sec2md==0.1.22` (solo dev, alpha, no tagged releases — explicit upgrade control).
