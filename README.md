# Bartleby, the Scrivener: A Tool of Wall Street

An AI-powered tool for processing document corpora and researching them with an agentic assistant.

Made with love by [John West](https://github.com/jswest), [Brian Whitton](https://github.com/noslouch), and [Rob Barry](https://github.com/robbarry).

---

## Background

At the _Wall Street Journal_, we have found it useful to let an AI agent run wild in a SQLite database containing the extracted text from a bunch of documents. Bartleby is the toolkit for that.

It's split into two pieces that share a SQLite database:

- **The `bartleby` CLI** scribes (parses, chunks, embeds, and indexes) documents. It also exposes helper commands that agents use during research sessions. Run on its own, it gives you a rich, queryable corpus regardless of whether you ever point an agent at it.
- **The `bartleby` skill** (in [`./skill`](./skill)) is a skill you drop into Claude Code, Cowork, or another compliant agent harness. It tells your agent how to explore the database, save findings, and cite evidence. The skill is BYO-model: it works with any agent the harness supports.

A SQLite database binds these two together. The CLI writes it, the skill romps through it, writing findings back into it as it cavorts.

A couple things to be aware of:

- Token costs can add up. For ingestion, summarization is the main driver (you can also turn it off). For research, costs are governed by whatever model you're running the skill against.
- This uses the excellent (but pre-v0) [`sqlite-vec`](https://github.com/asg017/sqlite-vec) plugin for SQLite. There might be some instability there.

---

## Installation

### Prerequisites

```
brew install uv
```

That's it. Bartleby uses [Docling](https://docling-project.github.io/docling/) for document conversion, which bundles OCR and structural parsing — no separate Tesseract or Playwright install required.

### Install Bartleby

From the project directory:

```
uv tool install .
```

This installs `bartleby` as a command-line tool in an isolated environment.

For development:

```
uv tool install --editable .
```

### Install the skill

The skill lives in [`./skill`](./skill). Copy it into your harness's skills directory. For Claude Code, that's typically:

```
cp -r skill ~/.claude/skills/bartleby
```

See [`./skill/README.md`](./skill/README.md) for harness-specific notes.

### A note on first-run latency

The first time you run `bartleby scribe`, it will pause for several minutes while it downloads:

- the Docling layout models (PDF table/structure detection, OCR weights),
- the `BAAI/bge-base-en-v1.5` embedding model (~400 MB),
- and the tokenizer assets that ride alongside both.

These are cached under `~/.cache/` and reused on every subsequent run, so the second ingest starts immediately. The first invocation of the skill's `search` script has a similar one-time wait for the embedding model when it loads in a fresh process.

If you want to warm the caches before your first real ingest, run `bartleby embed "warm up"` once — that loads BGE — and `bartleby scribe --files <one small pdf>` once, which loads Docling.

---

## Quick start

### 1. Configure

```
bartleby ready
```

The setup wizard asks for LLM provider/model, API keys, summary depth, temperature, and the max token threshold for reading whole documents. Settings save to `~/.bartleby/config.yaml`.

### 2. Create a project

```
bartleby project create foo
```

This creates a project directory (foo in this case) and marks it active. Subsequent commands use the active project unless you pass `--project`.

### 3. Ingest documents

```
bartleby scribe --files /path/to/your/docs
```

Point this at a file or directory of `.pdf`, `.html`, `.md`, or `.txt` files. Bartleby extracts text, chunks it, generates embeddings, and (optionally) writes a one-shot summary per document. Everything goes into the project's SQLite database.

### 4. Start an agent session

In your harness of choice, load the `bartleby` skill and ask the agent a question about your corpus. The skill will guide it through searching, reading, synthesizing, and citing.

If you want the agent to ignore findings from prior sessions, start the session with the memory flag off:

```
bartleby session start --no-memory
```

(More on sessions and memory in the skill README.)

---

## Architecture

```
                       ┌──────────────────┐
                       │  bartleby ready  │
                       │  bartleby scribe │   ← you, at the terminal
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  SQLite database │
                       │   (the contract) │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  bartleby skill  │   ← your agent, via your harness
                       │  search / read / │
                       │  save / cite     │
                       └──────────────────┘
```

The CLI owns ingestion. The skill owns research. The database is the API between them. Each piece can be replaced independently as long as the schema contract holds.

The database is self-describing — schema version, embedding model, and `sqlite-vec` version live in a `meta` table inside the DB itself. The skill reads `meta` on startup and refuses to run against an incompatible database.

---

## Project directory structure

```
~/.bartleby/projects/<name>/
├── bartleby.db       # everything: chunks, summaries, findings, sessions, audit log
└── archive/          # original PDF files, deduplicated by content hash
```

All queryable state lives in `bartleby.db`. Findings, audit logs, and agent-generated summaries are all stored as rows there — no sidecar files, no on-disk reports.

---

## Command reference

### `bartleby ready`

Interactive configuration wizard. Asks for:

| Setting | Default | Description |
| --- | --- | --- |
| LLM provider | anthropic | `anthropic`, `openai`, or `ollama` |
| Model | varies by provider | Model name (e.g., `claude-haiku-4-5`, `gpt-5-mini`, `gpt-oss:20b`) |
| API key | — | Required for Anthropic/OpenAI; can also use env vars |
| Summary depth | `one-shot` | `none` or `one-shot` |
| Temperature | 0 | 0 = deterministic, 1 = creative |
| Max summarize tokens | 50000 | If a document exceeds this, only the first N tokens are summarized (with a note appended) |
| Max read tokens | 50000 | Threshold above which the skill's `read_document` requires `--force` |

**API keys** can be provided in the config or via environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. For Ollama, configure the server URL (default `http://localhost:11434`) or set `OLLAMA_API_BASE`.

**A note on Ollama defaults.** The default Ollama model (`gpt-oss:20b`) assumes you have ~16GB of GPU/unified memory free. If you're on smaller hardware, override it during `bartleby ready` — anything in the Ollama library that handles summarization well will work. The "right" local default is determined by your machine, not by us.

**A note on token counts.** `documents.token_count` is computed with `tiktoken`'s `cl100k_base` encoder regardless of the LLM provider you're using. It's a rough estimate — accurate enough for the `read_document --force` gate, but not authoritative across providers.

Config saves to `~/.bartleby/config.yaml`.

### `bartleby project`

Manage project workspaces. Each project gets its own database and document archive.

```
bartleby project create <name>    # Create and activate a new project
bartleby project list             # List all projects
bartleby project use <name>       # Switch active project
bartleby project info [name]      # Show project details
bartleby project delete <name>    # Delete a project and all its data (-y to skip prompt)
```

### `bartleby scribe`

Ingest HTML, MD, PDF, and TXT documents into the project database. (Previously named `bartleby read` — renamed in v1 because the scribe writes the corpus; the agent reads it.)

```
bartleby scribe --files <path> [options]
```

| Option | Description |
| --- | --- |
| `--files <path>` | Path to a file or directory of supported documents (required) |
| `--project <name>` | Target project (defaults to active) |
| `--model <name>` | Override LLM model for summarization |
| `--provider <name>` | Override LLM provider |
| `--verbose` | Show debug output |

**Supported file types:** `.pdf`, `.html`/`.htm`, `.md`, `.txt`.

Ingestion runs sequentially. The embedding and Docling ML models are heavy, and small corpora don't benefit enough from parallelism to justify the warmup cost and complexity.

**Pipeline:**

1. Hashes and archives the source file at `archive/<hash>/<hash>.<ext>` (dedup by content).
2. Converts and chunks:
   - `.pdf`, `.html`, `.md`: [Docling](https://docling-project.github.io/docling/) — layout-aware, structural, with internal OCR for image-based PDFs. Chunks carry `section_heading` and `content_type`.
   - `.txt`: read as UTF-8, simple character chunker (Docling has no text reader).
3. Computes a `tiktoken` token count for the document.
4. Generates vector embeddings (BAAI/bge-base-en-v1.5, 768 dims).
5. Generates a one-shot, whole-document summary per document (if summary depth is `one-shot`). The summarizer enforces structured JSON output across all providers (anthropic, openai, ollama) via Pydantic — useful when an open-source model might otherwise drift into "Here's your summary:" preambles.
6. For documents longer than `max_summarize_tokens`, the summarizer runs on the first N tokens only and a deterministic note is appended to the saved summary.
7. Stores everything in SQLite with full-text search (FTS5) and vector search (sqlite-vec).

### `bartleby session`

Manage agent sessions. Sessions are first-class rows in the database; findings and audit log entries are tagged with a `session_id`.

```
bartleby session start [--no-memory]   # Start a new session, print its ID and name
bartleby session current               # Show the active session
bartleby session end                   # End the active session (cosmetic; sessions don't really "end")
```

**Most users will never run `bartleby session start`.** If no session is active when the skill calls a script, the skill auto-creates one with default settings (memory on). You only need to start a session explicitly if you want `--no-memory`.

The `--no-memory` flag creates a session that cannot read findings from prior sessions. This is enforced at the script level — the skill's `search` script returns no findings when called against a memory-off session, regardless of how the agent is prompted.

### `bartleby embed`

Embed a string and print the resulting vector as JSON. Used by the skill's `search` script during semantic search; rarely called directly.

```
bartleby embed "your query here"
```

### `bartleby logs`

View the audit log for a session. Useful when an agent does something weird and you want to see what tools it called.

```
bartleby logs [--session <name>] [--limit <n>]
```

If no session is specified, shows the most recent session's logs.

---

## Supported LLM providers (for ingest summarization)

| Provider | Default model | Notes |
| --- | --- | --- |
| Anthropic | `claude-haiku-4-5` | Requires API key. Structured output via tool-use. |
| OpenAI | `gpt-5-mini` | Requires API key. Structured output via the SDK's Pydantic parse helper. |
| Ollama | `gpt-oss:20b` | Local server. Structured output via the chat API's `format=` JSON schema. Pick a smaller model on smaller hardware. |

Summarization is text-only in v1 — we pass the document's extracted text to the model, not images. These providers govern summarization at ingest only. Research is whatever model your harness is running the `bartleby` skill against.

---

## Running fully local (for sensitive work)

Bartleby is built to run end-to-end without an internet connection. This is the path for journalists working with sensitive material:

1. **Ingest** — Run `bartleby ready`, set `provider: ollama`, and pick a model your hardware can run. Ingestion, embeddings, and summarization stay on the machine.
2. **Research** — Install [Goose](https://goose-docs.ai/) (Block's open-source agent, Apache 2.0) and point it at the same local Ollama in its provider settings. Goose reads Anthropic's Agent Skills format from `~/.claude/skills/`, so the `cp -r skill ~/.claude/skills/bartleby` install you'd do for Claude Code works unchanged.

No prompts, source text, or research notes leave the machine.

**A note on model quality.** Local models follow tool-use protocols less reliably than frontier cloud models. Bartleby's research loop (search → read → cite → save) asks the model to track `chunk_id`s and cite them accurately; smaller models sometimes drop or hallucinate them.

| Hardware | What to expect |
| --- | --- |
| ~16 GB unified memory | `gpt-oss:20b` or `llama3.1:8b` works; spot-check the agent's citations. |
| 32 GB+ | `llama3.3:70b` or `qwen2.5:32b` quantized — closer to trustworthy citation discipline. |
| Under 16 GB | Either run ingest locally and use a cloud model for research, or accept noticeably worse citations. |

The tradeoff between fully-local and cloud-research-with-local-ingest is real: privacy vs. citation reliability. Pick by project.

---

## What's the `bartleby` skill?

A folder you drop into your agent harness that teaches the agent how to use this database. It exposes a small set of scripts (`search`, `read_document`, `save_finding`, etc.) and a `SKILL.md` that codifies an opinionated research methodology — what counts as evidence, when to read a full document vs. searching, how to cite.

See [`./skill/README.md`](./skill/README.md) for the full story.

---

## License

MIT.

---

## Anything else?

"Ah Bartleby! Ah Humanity!"