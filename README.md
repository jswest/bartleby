# Bartleby, the Scrivener: A Tool of Wall Street

```
 ██████╗  █████╗ ██████╗ ████████╗██╗     ███████╗██████╗ ██╗   ██╗
 ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║     ██╔════╝██╔══██╗╚██╗ ██╔╝
 ██████╔╝███████║██████╔╝   ██║   ██║     █████╗  ██████╔╝  ╚████╔╝
 ██╔══██╗██╔══██║██╔══██╗   ██║   ██║     ██╔══╝  ██╔══██╗   ╚██╔╝
 ██████╔╝██║  ██║██║  ██║   ██║   ███████╗███████╗██████╔╝    ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝╚═════╝     ╚═╝
```

An AI-powered tool for processing document corpora and researching them with an agentic assistant--or in other words: Bartleby is a scrivener who might prefer not to. Made with love by [John West](https://github.com/jswest), [Brian Whitton](https://github.com/noslouch), and [Rob Barry](https://github.com/robbarry).

---

## Background

At the _Wall Street Journal_, we have found it useful to let an AI agent run wild in a SQLite database containing the extracted text from a bunch of documents. Bartleby is the toolkit for that.

It's split into two pieces that share a SQLite database:

- **The `bartleby` CLI** scribes (parses, chunks, embeds, and indexes) documents. It also exposes helper commands that agents use during research sessions. Run on its own, it gives you a rich, queryable corpus regardless of whether you ever point an agent at it.
- **The `bartleby` skill** (in [`./bartleby/skill`](./bartleby/skill)) is a skill you drop into Claude Code, Cowork, Goose, or another compliant agent harness. It tells your agent how to explore the database, save findings, and cite evidence. The skill is BYO-model: it works with any agent the harness supports — full story in its [README](./bartleby/skill/README.md).

A SQLite database binds these two together. The CLI writes it, the skill romps through it, writing findings back into it as it cavorts.

A couple things to be aware of:

- Token costs can add up. For ingestion, summarization and image description are the drivers (you can also turn either off or use local models). For research, costs are governed by whatever model you're running the skill against. If your hardware supports it, you can run everything locally, though (see below).
- This uses the excellent (but pre-v0) [`sqlite-vec`](https://github.com/asg017/sqlite-vec) plugin for SQLite. There might be some instability there.

---

## Install and update

### Prerequisites

```
brew install uv tesseract
```

(`apt install tesseract-ocr` on Debian/Ubuntu; on Windows, use the official installer from UB Mannheim.)

Tesseract does cheap OCR on scanned PDF pages before falling back to the more expensive VLM. The default PDF pipeline uses [pdfplumber](https://github.com/jsvine/pdfplumber) for text and [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) for page rendering — both are bundled as Python deps, no system install needed.

Both paths below install with the `docling` and `sec2md` extras: [Docling](https://docling-project.github.io/docling/) is the layout-aware converter, required to ingest `.md`/`.html` files, so you almost always want it; [sec2md](https://github.com/alphanome-ai/sec2md) is a specialist for iXBRL EDGAR filings. A bare `uv tool install .` works but can't ingest those formats. (Development: `uv tool install --editable .`.)

**WSJ-internal users** who want the wsjpt provider need one extra flag on the install/update commands below — see [`docs/wsj-internal.md`](./docs/wsj-internal.md).

Both paths end with `bartleby ready`; restart your harness afterward so it reloads the skill.

### Riding `main`

**Install:**

```
git clone https://github.com/jswest/bartleby.git   # skip if you already have the repo
cd bartleby
uv tool install '.[docling,sec2md]' --force
bartleby ready
```

**Update** — pull, then repeat the same two commands:

```
git pull
uv tool install '.[docling,sec2md]' --force
bartleby ready
```

Editable installs (`--editable .` in place of the install command) pick up plain code changes automatically — skip reinstalling for those, but still re-run `bartleby ready` (`--check` reports whether anything actually changed).

**If a `git pull` added or bumped a dependency**, reinstall even on an editable install — it only references your source tree, so it won't pick up anything newly added to `pyproject.toml`. The symptom is a stray `ModuleNotFoundError` from a command that used to work fine ([#697](https://github.com/jswest/bartleby/issues/697)); fix it with `uv tool install --reinstall '.[docling,sec2md]'`.

**Verify:**

```
which bartleby                          # the CLI is on PATH
bartleby --version                      # which version (or dev build) is installed
bartleby project list                   # the CLI actually runs
bartleby ready --check                  # the installed skill is present and current
```

### Pinned release

Releases are git tags of the form `v0.<schema>.<patch>` — the **minor** number *is* the database schema version. Same minor → a safe in-place upgrade; a higher minor → the schema changed and existing corpora need [`bartleby project upgrade`](#after-a-schema-change) or a re-ingest. (Maintainers: see [`scripts/release.py`](./scripts/release.py) for how tags are cut.)

**Install** — browse the [releases page](https://github.com/jswest/bartleby/releases) or list tags without cloning, then pin to one:

```
git ls-remote --tags https://github.com/jswest/bartleby.git 'v*'
uv tool install 'git+https://github.com/jswest/bartleby.git@v0.7.0#egg=bartleby[docling,sec2md]'
bartleby ready
```

The `#egg=bartleby[...]` fragment is how extras attach to a `git+https` URL — drop it and you get a working CLI with no Docling/sec2md, which silently breaks HTML/EDGAR ingestion. `bartleby --version` always reports exactly what tag you're running.

**Update** — find the latest tag the same way, compare its minor to your current `bartleby --version`, then reinstall at the new tag with `--force`:

```
uv tool install 'git+https://github.com/jswest/bartleby.git@<latest>#egg=bartleby[docling,sec2md]' --force
bartleby ready
```

**Verify:** same four commands as under riding `main`, above.

### After a schema change

Whichever path you're on, a schema bump makes an existing project fail to open with a `schema version mismatch` error. Bring it up to date:

```
bartleby project upgrade <name>
```

Most updates upgrade in place; when a change isn't backward-compatible, `upgrade` tells you to re-ingest instead (recreate the project and run `bartleby scribe` again) — there's no automatic migration for those.

**If you have findings older than the `[^chunk:N]` citation format** ([#624](https://github.com/jswest/bartleby/issues/624)): an old-style marker doesn't error, it just stops being recognized, so `finding_citations` can go stale with no signal anything broke. A one-time backfill already fixed every corpus present when it ran ([#642](https://github.com/jswest/bartleby/issues/642)), but a corpus adopted from elsewhere can still carry one. `bartleby project upgrade` won't touch this — it's a data issue, not a schema one. Fix it by rewriting the finding's body through `edit_finding` (see the [skill reference](./bartleby/skill/README.md)), which re-extracts citations under the current grammar.

### Gotchas

- Don't keep the repo (or its `.venv`) in a synced folder like Dropbox, iCloud, or OneDrive — syncing rewrites file paths and quietly breaks the install.
- `bartleby` isn't on PyPI: don't `uv pip install bartleby` or `uvx bartleby`. Riding `main`, run the `uv` commands from inside the repo checkout (a pinned install runs from anywhere).

---

## Quick start

### 1. Configure

```
bartleby config
```

The setup wizard asks for LLM provider/model, API keys, summary depth, temperature, and the max token threshold for reading whole documents. Settings save to `~/.bartleby/config.yaml`.

![bartleby config: the interactive setup wizard walking through provider, model, and summarization settings.](./docs/demo.gif)

### 2. Create a project

```
bartleby project create foo
```

This creates a project directory (`foo` in this case) and marks it active. Subsequent commands use the active project unless you pass `--project`.

### 3. Ingest documents

```
bartleby scribe --files /path/to/your/docs
```

Point this at a file or directory of `.pdf`, `.html`, `.md`, `.txt`, or image files (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`); unrecognized extensions are content-sniffed and kept if they turn out to be a supported type (details under [`bartleby scribe`](#bartleby-scribe)). Bartleby extracts text, chunks it, generates embeddings, and (optionally) writes a one-shot summary per document. With a vision provider configured, embedded images and standalone image files are analyzed too (OCR + scene description) and folded into the same searchable index. Everything lands in the project's SQLite database.

### 4. Start an agent session

In your harness of choice, load the `bartleby` skill (install it first with [`bartleby ready`](#bartleby-ready)) and ask the agent a question about your corpus. The skill guides it through searching, reading, synthesizing, and citing.

The agent opens its own research *run* on its first call (`bartleby skill session new`) and carries it for the rest of the conversation, so **one conversation is one run** with no setup on your part — a new conversation is a new run.

**Memory off.** To have the agent ignore findings from prior runs (e.g. for a blind multi-model comparison), say so in the *first message* of a new conversation — "ignore previous memory" works. The agent then opens its run with:

```
bartleby skill session new --no-memory
```

Memory-off is fixed when the run opens — ask mid-conversation and the agent will tell you to start a new one. (`bartleby session start --no-memory` doesn't do it; more in the [skill README](bartleby/skill/README.md#how-runs-work).)

### 5. Browse what you've got

Run `bartleby serve` for a local web UI over the corpus and findings — see [`bartleby serve`](#bartleby-serve).

### 6. Share a single finding out of band

```
bartleby finding read <finding-id>              # render to stdout as Markdown (--json, --render)
bartleby finding export <finding-id>            # writes <slug>.md (or pass --out PATH)
bartleby finding import path/to/finding.md      # into the active project (or --project)
```

Details under [`bartleby finding`](#bartleby-finding).

---

## Architecture

The CLI ingests. The skill researches. The database is the API between them, so either side can be replaced as long as the schema contract holds. The DB is self-describing: schema version, embedding model, and `sqlite-vec` version live in its `meta` table, and Bartleby refuses to open an incompatible one.

- **Schema:** [`bartleby/db/schema.py`](./bartleby/db/schema.py) is the DDL and the source of truth for tables.
- **Invariants and current state:** [`ARCHITECTURE.md`](./ARCHITECTURE.md) — the polymorphic `chunks` table, the single-writer ingest pipeline, memory-off enforcement, flag and id conventions.
- **Why past calls went the way they did:** [`docs/decisions/`](./docs/decisions/).

The one thing worth knowing up front: documents, summaries, findings, and images all land in one polymorphic `chunks` table (shadowed by FTS5 and `sqlite-vec` indexes), so a single search covers every kind of source at once.

Tag and finding curation lives on the skill surface, not the CLI: `bartleby skill <name>` (e.g. `bartleby skill add_tag`, `bartleby skill assign_tag`, `bartleby skill save_finding`) is the sanctioned human path for managing tags and findings.

---

## Project directory structure

```
~/.bartleby/projects/<name>/
├── bartleby.db       # everything: chunks, summaries, findings, sessions, audit log, images
└── archive/          # original document files, dedup'd by content hash
    ├── <doc_hash>/<doc_hash>.<ext>
    └── images/<img_hash>.jpg   # extracted figures, scanned page renders, standalone images
```

All queryable state lives in `bartleby.db`. Findings, audit logs, and agent-generated summaries are all stored as rows there — no sidecar files, no on-disk reports.

Set `BARTLEBY_HOME` to relocate this whole tree — `projects/`, `config.yaml`, and scratch — somewhere other than `~/.bartleby`. Useful for keeping more than one corpus root, for CI, or for sandboxing a tool/agent so it can't touch your live corpora.

---

## Command reference

### `bartleby config`

Interactive configuration wizard. Asks for:

| Setting | Default | Description |
| --- | --- | --- |
| LLM provider | anthropic | `anthropic`, `openai`, or `ollama` (plus `wsjpt`, WSJ-internal) |
| Model | varies by provider | Model name (e.g., `claude-haiku-4-5`, `gpt-5-mini`, `qwen3-vl:30b`) |
| API key | — | Required for Anthropic/OpenAI; can also use env vars |
| Summary depth | `one-shot` | `none` or `one-shot` |
| Temperature | 0 | 0 = deterministic, 1 = creative |
| Reasoning effort | `low` | `minimal`/`low`/`medium`/`high`; only prompted when summary depth is `one-shot` |
| Max summarize tokens | 50000 | Documents over this length are summarized from the first N tokens, with a note appended |
| Summarize workers | 4 (cloud) / 1 (Ollama) | How many documents summarize in parallel after parsing; Ollama auto-clamps to 1 (not prompted) |
| PDF converter | `pdfplumber` | `pdfplumber` (fast, default) or `docling` (slower, more structurally aware) |
| HTML converter | `docling` | `docling` (default; also handles `.md`) or `sec2md` (routes iXBRL EDGAR filings to sec2md, other HTML to docling) |
| Sparse-text threshold | 100 | Pages with fewer extracted chars are treated as scanned; OCR then VLM fallback |
| Parse workers | auto | How many documents to parse in parallel; `0` auto-sizes to `min(CPU cores − 2, free RAM ÷ 12 GB)`, at least 1 |
| Vision provider | (off) | Off by default; opt in during the wizard. If enabled, choose `anthropic`, `openai`, or `ollama` (plus `wsjpt`, WSJ-internal) |
| Vision model | varies by provider | e.g., `claude-haiku-4-5`, `gpt-5-mini`, `qwen3-vl:30b` |
| Max image dimension | 768 | Long-edge pixels before sending an image to the VLM |
| Min image dimension | 64 | Images with a shorter edge than this are skipped — avoids wasted VLM calls (and crashes) on thin slivers |
| Tesseract min confidence | 30 | Avg confidence (0-100) below which we fall back to the VLM on sparse pages |
| Caption workers | 4 (cloud) / 1 (Ollama) | How many images caption in parallel after parsing; Ollama auto-clamps to 1 (not prompted) |
| Max read tokens | 50000 | Threshold above which the skill's `read_document` requires `--force` |

Reasoning effort trades billed tokens for depth (OpenAI gpt-5 and effort-capable Anthropic models only; Ollama/wsjpt ignore it). Summarize and caption workers run network-bound LLM/VLM calls as their own pipeline stage; the Ollama clamp is because `OLLAMA_NUM_PARALLEL` defaults to 1, so parallel requests only queue. Parse workers are RAM-bound and recycle periodically (see [ARCHITECTURE.md](./ARCHITECTURE.md#single-writer-drain--per-unit-resume)); a count you set explicitly can use every core.

**API keys** can be provided in the config or via environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (optional for wsjpt — it defaults to Vertex AI / ADC; setting this, or `wsjpt_api_key` in config, switches it to the Gemini API-key path). For Ollama, configure the server URL (default `http://localhost:11434`) or set `OLLAMA_API_BASE`.

For local-only setups, see [Running fully local](#running-fully-local-for-sensitive-work) for the recommended model picks by hardware tier.

Config saves to `~/.bartleby/config.yaml`.

### `bartleby ready`

Install or refresh the skill into your agent harness. Stamps the skill bundled with your installed `bartleby` into `~/.claude/skills/bartleby/` (or `--dest <dir>`), replacing any prior copy so `SKILL.md` lands directly under it. "Latest" is decided by a content hash over the skill files, not the version number, so re-running is a no-op only when the installed copy already matches.

| Flag | Effect |
| --- | --- |
| (none) | Install or refresh if the installed copy differs from the bundled one; no-op if already current. |
| `--check` | Report status and exit non-zero if missing or stale; writes nothing. |
| `--force` | Reinstall even when already up to date. |
| `--dest <dir>` | Install into a different skill directory. |

Restart your harness afterward — skills load at startup.

### `bartleby project`

Manage project workspaces. Each project gets its own database and document archive.

```
bartleby project create <name>                     # Create and activate a new project
bartleby project list                              # List all projects
bartleby project use <name>                        # Switch active project
bartleby project info [name]                       # Show project details (--verify for integrity checks)
bartleby project delete <name>                     # Delete a project and all its data (--yes to skip prompt)
bartleby project upgrade <name>                    # Apply additive schema upgrades to an existing DB
bartleby project publish <name> --to <s3-url>      # Publish a findings-free copy (+ originals) to S3
bartleby project import <name> --from <source>     # Import a published corpus as a new local project
```

`publish` strips findings and sessions from a copy of the corpus before uploading the `.db` and archived originals to an S3 prefix. `import` pulls one back down (`s3://…`, a local directory, or `file://…`) as a brand-new project — refusing on a schema or embedding-model mismatch — optionally dropping tags (`--without-tags`) or overwriting a same-named project (`--yes`, which drops its local findings).

### `bartleby finding`

Read, export, or import a single finding out of band.

```
bartleby finding read <finding-id>              # render to stdout as Markdown (--json, --render)
bartleby finding export <finding-id>            # writes <slug>.md (or pass --out PATH)
bartleby finding import path/to/finding.md      # into the active project (or --project)
```

- `finding read` is the read-only, terminal-facing companion: it renders one finding to stdout as Markdown — title, provenance subtitle, and the body with citations resolved *live against the current corpus* into numbered footnotes (`† file · p.N`, `‡ source no longer available`, `§` for external refs). Pipe it to a pager (`| less`, `| glow`) or pass `--render` to pretty-print in place; `--json` emits the raw finding. It resolves live rather than baking inert markers, so it's for reading here, not sharing elsewhere.
- `finding export` writes a self-describing Markdown artifact: YAML front matter (title, description, source corpus, original finding id, export date) followed by the body, with corpus citations rewritten as inert `[corpus: <file> · p.<N>]` markers so it stands alone without the corpus.
- `finding import` parses such an artifact into a project through the normal finding write path, prepending the provenance as a header line. Imported citations stay inert — never re-resolved to local chunk ids — and the finding then renders like any other local one.

Together, `export`/`import` are the lightweight, no-S3 alternative to `bartleby project publish`/`import` for handing off one finding. For a fully *rendered* hand-off — the web view itself, with fonts and cited sources embedded in one HTML file — use the **Save as HTML** button in [`bartleby serve`](#bartleby-serve) instead.

### `bartleby scribe`

Ingest HTML, MD, PDF, and TXT documents into the project database.

```
bartleby scribe --files <path> [<path> ...] [options]
```

| Option | Description |
| --- | --- |
| `--files <path> [<path> ...]` | One or more files and/or directories of supported documents (required). Directories are walked recursively; a file reachable from more than one path is ingested once. |
| `--only <type>` | Restrict ingestion to the given file type(s): `pdf`, `html`, `md`, `txt`, `image`. Repeatable and/or comma-separated (e.g. `--only pdf,html`). Filters on the *resolved* type, so a content-sniffed PDF with no extension is kept by `--only pdf`. |
| `--project <name>` | Target project (defaults to active) |
| `--model <name>` | Override LLM model for summarization |
| `--provider <name>` | Override LLM provider |
| `--pdf-converter <name>` | Override PDF converter (`pdfplumber` or `docling`) |
| `--html-converter <name>` | Override HTML converter (`docling` or `sec2md`) |
| `--verbose` | Show debug output |
| `--timings` | Benchmark mode: time each document's parse/embed/caption/summarize stages, print the per-doc split to stderr, and emit an aggregate (docs/sec, pages/sec, per-stage breakdown) as JSON to stdout. Off by default — normal ingest is unchanged. |

**Supported file types:** `.pdf`, `.html`/`.htm`, `.md`, `.txt`, image files (`.jpg`/`.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`/`.tif`). The type is taken from the extension when it is one of these; a missing or unrecognized extension is resolved by sniffing the file's magic bytes instead. A recognized extension is always trusted as-is — content never overrides it — so a `.txt` that happens to hold PDF bytes stays text.

Ingestion runs in three concurrent phases — parse (a process pool), image caption, and summarize (each its own worker pool, sized by the wizard settings above) — all feeding a single writer that owns the database connection. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the single-writer drain and how a run resumes by what's missing.

**Pipeline:**

1. Hashes and archives the source file at `archive/<hash>/<hash>.<ext>` (dedup by content).
2. Converts and chunks:
   - `.pdf`: pdfplumber by default — per-page text extraction; embedded images are extracted via page-render-crop. Pages whose extracted text is below `sparse_text_threshold` are treated as scanned: Tesseract OCR runs first (cheap), and only if confidence is below `ocr_min_confidence` does the page get routed to the VLM.
   - `.pdf` with `--pdf-converter docling`: layout-aware, structural extraction with internal OCR for image-based PDFs.
   - `.html`, `.htm`, `.md`: Docling by default (requires the `[docling]` install). With `--html-converter sec2md`, each HTML file is sniffed for the iXBRL namespace — matches route to sec2md (preserves SEC tables + section headings); non-matches still go through Docling. `.md` always goes through Docling.
   - `.txt`: read as UTF-8, simple character chunker — *unless* it's an EDGAR full-submission file (detected by its `<SEC-DOCUMENT>`/`<SEC-HEADER>` SGML envelope). Those are unwrapped into inner `<DOCUMENT>` blocks: HTML/iXBRL bodies route to sec2md (requires `[sec2md]` — the only converter that reads SEC HTML), plain-text exhibits use the character chunker, graphics/XBRL files are skipped, and the whole submission lands as one document. This overrides `html_converter` for inner HTML only; standalone EDGAR `.htm` files still honor it.
   - Image files: routed directly to the VLM. OCR transcription and scene description are stored as separate chunks (`content_type='image_ocr'` and `'image_description'`).
3. Computes a `tiktoken` token count for the document.
4. Generates vector embeddings (BAAI/bge-base-en-v1.5, 768 dims).
5. Generates a one-shot, whole-document summary per document (if summary depth is `one-shot`). The summarizer enforces structured JSON output across all providers (anthropic, openai, ollama) via Pydantic. The same call also extracts an optional `authored_date` (ISO 8601) if the document states one; malformed or ambiguous dates store as NULL.
6. For documents longer than `max_summarize_tokens`, the summarizer runs on the first N tokens only and a deterministic note is appended to the saved summary.
7. Stores everything in SQLite with full-text search (FTS5) and vector search (sqlite-vec). Images dedupe at the byte level — the same icon embedded in five docs is one VLM call, not five.

**Ingest is restartable:**

- Each document's parse, each image caption, and the summary commit as independent units — a crash, Ctrl-C, or a VLM outage mid-corpus loses no completed work.
- Re-running the same command resumes by what's *missing*: unfinished images get re-captioned, finished work is never redone, and a fully-ingested file is skipped.
- A unit that keeps failing is retried a few times, then left out and reported rather than retried forever — counted under "Failed units" in `bartleby project info`.
- A run exits non-zero if any unit is left unresolved, so a scripted caller (`bartleby scribe ... && next-step`) can trust the exit code; a fully successful run, including a no-op resume, exits 0.

**Backfilling dates.** `bartleby scribe backfill-dates [project] --from-filename '<regex>'` bulk-sets `authored_date` from a named `date` capture group matched against each file's name (`--match-path` matches the full path instead). Fills `NULL`s only unless `--overwrite`; `--dry-run` reports counts and sample matches without writing. A human-run admin op, not on the skill's surface — the skill's `probe_dates` validates a regex and hands you the exact command to run.

**Benchmarking ingest.** `--timings` turns the run into a repeatable measurement, timing each document's stages and emitting a per-stage aggregate as JSON to stdout (the bar and prose stay on stderr, so capture it with a redirect). The reproducible recipe — fresh-project setup, the aggregate JSON field reference, the gotchas that silently corrupt a run, recorded runs, and [rough per-document expectations](docs/BENCHMARKS.md#rough-expectations-anecdotal) — lives in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### `bartleby session`

Inspect and label research runs from your side. A run (a *session* in the database) is a row that findings and audit log entries are tagged with. **Agents open their own** — one per conversation, via `bartleby skill session new` (see [Quick start step 4](#4-start-an-agent-session)) — so you don't need to start one before pointing an agent at the corpus.

```
bartleby session start [--no-memory] [--harness <name>] [--model <id>]   # Start a session and mark it active
bartleby session current                                                 # Show the active session
bartleby session end                                                     # End the active session (cosmetic)
bartleby session set [--harness <name>] [--model <id>]                   # Stamp the active session's backend
```

The *active* session is whichever was started or used most recently — usually the last agent run, since every agent call re-marks its run active. So `current` shows the latest run, and `set` stamps it: for a blind multi-model comparison, let the agent run without `--model`, then `session set --model <id>` after assessing. `--harness` is best-effort auto-detected (e.g. Claude Code) when omitted; `--model` usually has no environment signal. Unknown values stay null — never guessed. The values show up in `list_findings` / `read_finding`.

`session start --no-memory` doesn't make an agent's run memory-off — the agent mints its own run on its first call and moves the active marker to it. See [Quick start step 4](#4-start-an-agent-session).

### `bartleby embed`

Embed a string and print the resulting vector as JSON. Used by the skill's `search` script during semantic search; rarely called directly.

```
bartleby embed "your query here"
```

### `bartleby logs`

View the audit log for a session. Useful when an agent does something weird and you want to see what tools it called.

```
bartleby logs [--session <name>] [--limit <n>] [--project <name>]
```

If no session is specified, shows the most recent session's logs. `--project` targets a project other than the active one; `--limit` defaults to 50.

### `bartleby serve`

Launch a local SvelteKit UI for browsing *and searching* the active project — findings, documents, and full corpus search, with inline citations that link straight into the archived PDFs at the right page.

```
bartleby serve
bartleby serve --project <name>   # browse a different corpus without switching the active one
```

Five top-level views (plus a per-chunk view reached from citations and search hits):

- `/` — a corpus overview for the active project (the same aggregate the agent's `describe_corpus` returns): document / chunk / token totals, the authored-date range shown with its undated count, a documents-by-year histogram, summary coverage, content mix, tag chips, and the largest documents — plus nav cards into findings and documents.
- `/search` — search the whole corpus using the same engine the agent uses. **Search** mode fuses full-text + semantic ranking (RRF) across documents, summaries, findings, and images; **Scan** mode enumerates *every* chunk matching a literal phrase, paginated. Filter by source kind, tag, and document scope; expand any hit to its full text or open the source file at the cited page. Each hit's `chunk N` carries a small open-in-context icon → its `/chunks/<id>` view. (Semantic queries load the embedding model per request, so the first hit takes a few seconds — the page shows a loading state.)
- `/findings` — every saved finding, newest first. Click through to a split view: the finding's body (markdown, with inline citation chips) on the left, the source PDF on the right. Clicking a chip jumps the viewer to the cited page; the small icon beside it opens that chunk's `/chunks/<id>` view. A **Save as HTML** button (beside *Copy as Markdown* / *Download .md*) downloads the finding as a single self-contained HTML file — fonts and every cited source embedded — that reproduces this view offline, on any machine, with no server and no network.
- `/documents` — the ingested corpus, filterable by authored-date range (with an include-undated toggle) and tag, sortable by title / date / ingest order, and paginated. Each row shows its assigned tag chips (hover a chip for the tag's description); when a date filter hides undated documents it says how many and offers to show them. Click through to a split view: the one-shot summary on the left, the original document on the right (PDFs in the browser's native viewer with `#page=` jumps; markdown rendered to formatted HTML; everything else in a sandboxed frame).
- `/tags` — the controlled tag vocabulary: every tag with its description and document count. Click a tag to see the documents carrying it.
- `/chunks/<id>` — a single chunk in context: the chunk itself at full contrast, its two neighbors on each side (same source, by chunk index) muted as surrounding context, and a link back to the source document (or finding). Reached from the icon beside any chunk reference in findings and search results.

Screenshots below are from a demo project built from NASA public-domain Hubble documents and imagery (1990–2009).

![Corpus overview (`/`) for the Hubble demo project: 3 findings and 25 documents, chunk and token totals, the authored-date range, documents by year, content mix, tags, and the largest documents.](./docs/serve-overview.png)

![Search (`/search`) for "astronaut spacewalk servicing the telescope": fused full-text + semantic hits led by two image descriptions, with source-kind, tag, and scope filters.](./docs/serve-search.png)

![A finding (`/findings/<id>`): "Why Hubble's mirror was flawed — and missed", with inline citations and margin source notes on the left and the cited Allen Report PDF open at page 4 on the right.](./docs/serve-findings.png)

![Documents (`/documents`): the Hubble corpus of PDFs, text files, and images as cards with file name, page count, tags, and one-shot summary, plus date, tag, and sort filters.](./docs/serve-documents.png)

![Tags (`/tags`): the controlled vocabulary — servicing-mission, investigation, press-kit, status-report, imagery, technical-paper — each with its description and document count.](./docs/serve-tags.png)

![A chunk in context (`/chunks/<id>`): a cited Allen Report passage at full contrast between its two muted neighbors on each side, with a link back to the source document.](./docs/serve-chunk.png)

Requires Node.js and npm on `PATH`; the first invocation runs `npm install` once into `~/.bartleby/serve/`.

- Opens the project database read-only, so it's safe to leave running alongside an ingest or a research session. The corpus overview, document listing, and search delegate to the skill scripts (`describe_corpus`, `list_documents`, `search`, `scan`, `read_chunks`) as subprocesses under a dedicated, memory-enabled `web-reader` session — so the views show exactly what the agent sees, and the web never disturbs whichever session an agent has active.
- Picks up the active project from `~/.bartleby/config.yaml`: `bartleby project use <name>` plus a page reload switches what you're looking at.

### `bartleby benchmark`

Pick the best local Ollama model for the document summarizer — and keep the
choice honest as your installed models change. A re-runnable selection tool:
`summarize` appends runs across every model × document, `judge` scores them
with a blind cloud judge, and `leaderboard` ranks the results; `blind` and
`errors` support spot-checking. Evidence accumulates in append-only stores, so
re-run any stage and the picture sharpens.

```
bartleby benchmark summarize   # append summarize runs (every model × document)
bartleby benchmark judge       # top up blind cloud-judge scores
bartleby benchmark leaderboard # the ranked report (--output writes CSV)
bartleby benchmark blind       # blinded summaries + key for a human spot-check
bartleby benchmark errors      # failed runs, with raw-output previews
```

Run from the repo root (or pass `--benchmarks-dir`). The full recipe — configs,
stores, and provenance — lives in [`benchmarks/README.md`](benchmarks/README.md).

---

## Supported LLM providers (for ingest summarization)

| Provider | Default LLM | Default VLM | Notes |
| --- | --- | --- | --- |
| Anthropic | `claude-haiku-4-5` | `claude-haiku-4-5` | Requires API key. Structured output via tool-use. |
| OpenAI | `gpt-5-mini` | `gpt-5-mini` | Requires API key. Structured output via the SDK's Pydantic parse helper. |
| Ollama | `qwen3-vl:30b` | `qwen3-vl:30b` | Local server. Structured output via the chat API's `format=` JSON schema. One MoE model handles both jobs; `gemma4:e2b` is a lighter alternative (see [Picking models for your hardware](#picking-models-for-your-hardware)). |
| wsjpt | `fast` | `fast` | WSJ-internal only — see [`docs/wsj-internal.md`](./docs/wsj-internal.md) for install and configuration. |

The same provider list is used for both ingest-time summarization (the LLM) and image analysis (the VLM). You can mix providers — e.g. OpenAI for summaries, local Ollama for image analysis — or run the same one for both. Research at the agent layer is governed by whatever model your harness is running the `bartleby` skill against, not by these settings.

## Tech stack

- **Storage:** SQLite with FTS5 (full-text) and [`sqlite-vec`](https://github.com/asg017/sqlite-vec) (vector). One file per project.
- **Embeddings:** [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) via `sentence-transformers`. 768 dimensions, ~400 MB on first download.
- **PDF text + image extraction:** pdfplumber (text per page, image bounding boxes), pypdfium2 (page rendering for OCR + image crops). Default converter.
- **OCR:** [Tesseract](https://tesseract-ocr.github.io/) via `pytesseract`. Cheap first pass for sparse pages.
- **VLM for image analysis:** pluggable — Anthropic / OpenAI / Ollama. Schema-enforced (Pydantic) JSON across providers, like the summarizer.
- **Converters:** Docling (default for HTML/MD, opt-in for PDF) and sec2md (Apache 2.0; iXBRL EDGAR HTML when opted in, required for HTML bodies inside EDGAR full-submission `.txt`) — see [Prerequisites](#prerequisites) and [`bartleby scribe`](#bartleby-scribe).
- **Token counting:** `documents.token_count` is computed with `tiktoken`'s `cl100k_base` encoder regardless of which LLM provider you're using. A rough estimate — accurate enough for the `read_document --force` gate, not authoritative across providers.

---

## Running fully local (for sensitive work)

Bartleby is built to run end-to-end without an internet connection — the path for journalists working with sensitive material. Two pieces, both pointed at the same local Ollama:

1. **Ingest** — Run `bartleby config`, set `provider: ollama` (and `vision_provider: ollama` if you want image analysis), and pick a model your hardware can run.
2. **Research** — Install [Goose](https://goose-docs.ai/) (Apache 2.0; originally Block's, now governed by the Linux Foundation's Agentic AI Foundation) and point it at the same local Ollama. Goose reads Anthropic's Agent Skills format from `~/.claude/skills/`, so the `bartleby ready` install you'd do for Claude Code works unchanged. If you have Ollama, you can run [Pi](https://pi.dev), which is also excellent with `ollama launch pi --model <model-slug>`.

No prompts, source text, or research notes leave the machine.

**How the models compare (as of June 2026).** We gave three models the same large, repetitive corpus and the same open-ended brief: surface accountability angles and save them as findings. The task rewards efficient search, claims grounded in a specific document, and judgment about what the evidence supports. We scored a frontier model (Claude Opus 4.8) against two locally-runnable open models (Qwen 3.6, 35B; Gemma 4, 31B) on time spent, factual accuracy, and editorial restraint.

- **Opus 4.8** worked longest but turned the time into dense, specific, well-cited output, and consistently separated what the documents proved from what would be an unfair leap.
- **Qwen 3.6** was a capable middle tier: real structural insight and usable leads, but a numeric error, some tool-use sloppiness, and one overreach asserting wrongdoing the filings didn't support.
- **Gemma 4** was fastest and shallowest — thematically plausible but light on specifics, with garbled citations and claims stated rather than shown. Buyer beware.

### Picking models for your hardware

As of June 2026:

| Hardware | Ingest (summarization and tagging) | Ingest (VLM) | Research (Goose or Pi) |
| --- | --- | --- | --- |
| 64 GB+ unified memory | `gpt-oss:120b` or `qwen3:30b` | `qwen3-vl:30b` | `gpt-oss:120b` or `qwen3.6:35b-mlx` |
| ~32 GB unified memory | `gpt-oss:20b` | `gemma4:e2b` (Can occasionally stall on structured-output JSON reparses, which shows up as an apparently slow run rather than an error.) | `gpt-oss:20b` |

`qwen3.6:35b-mlx` is fine for research but **not** for ingest summarization — it failed the summarizer's structured-JSON contract 10/10 in testing (`docs: 0`); see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#gotchas-that-silently-corrupt-a-run).

**A note on model quality.** Local models follow tool-use protocols less reliably than frontier cloud models. Bartleby's research loop (search → read → cite → save) asks the model to track `chunk_id`s and cite them accurately; smaller models sometimes drop or hallucinate them. They can also format them incorrectly, which is super annoying. `gpt-oss:120b` is reasonably disciplined; with `gpt-oss:20b` you'll want to spot-check.

If you can't fit either tier, the middle path is **local ingest + cloud research**: keep `provider: ollama` for the deterministic ingest pipeline, but point Goose or Pi (or Claude Code) at a frontier API for the agent layer. Source documents still never leave the machine; only the agent's queries do.

### Sandboxing Pi in an isolated VM

Pi is a minimal harness with an unsandboxed `bash` tool — handing a local model a shell on your machine. To keep that off your Mac, [`docs/pi-vm-runbook.md`](docs/pi-vm-runbook.md) walks through running Pi (plus `decant` for web fetches) inside an isolated Apple `container` VM that mounts only your corpus: the big agent model stays on the host GPU, `decant`'s small distill model runs CPU-only inside the box, and helper scripts live in [`scripts/pi-vm/`](scripts/pi-vm/).

### Model downloads and offline mode

Models download from the Hugging Face Hub lazily, the first time each is needed: the `BAAI/bge-base-en-v1.5` embedding model (~400 MB plus tokenizer assets) on your first `bartleby scribe` (or the skill's first `search`), and — when you ingest with Docling — its layout/OCR/table models on first use. They cache under `~/.cache/huggingface/hub` and download once.

To avoid a Hub network check on every run, Bartleby switches Hugging Face into offline mode automatically — but only once *every* model the current run needs is already cached. Until then it stays online so the missing model can download. If you ever hit a model fetch that's blocked by offline mode, re-run with `HF_HUB_OFFLINE=0` to force the download; an explicit `HF_HUB_OFFLINE` in your environment always overrides Bartleby's default.

---

## Contributing

Bartleby is built hand-in-hand with Claude Code, and the workflow that makes that
work — the `/ship` issue→PR loop (leaf issues and omnibus bundles alike), the
worktree convention, the commit gates, the safety hook — is version-controlled
right in the repo. See
[`CONTRIBUTING.md`](./CONTRIBUTING.md) for how we develop here (and how to do it by
hand if you'd rather). Architectural invariants live in
[`ARCHITECTURE.md`](./ARCHITECTURE.md); the decision log in [`docs/decisions/`](./docs/decisions/).

---

## License

MIT.

---

## Anything else?

"Ah Bartleby! Ah humanity!"