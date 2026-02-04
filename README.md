# Bartleby, the Scrivener

An AI-powered tool for processing document corpora and researching them with an agentic assistant.

---

## Background

I have found it useful to let an AI agent run wild in a SQLite database containing the extracted text from a bunch of documents. I've explored giving that agent various tools to explore the database more effectively, including full-text and semantic searching. This provides a toolkit _and_ agent to research and generate reports based on caches of PDF documents.

**`bartleby read`** handles the parsing side: OCR-ing and parsing PDFs (and converting HTML files) into a SQLite database, then paginating, summarizing, chunking, and embedding. This is valuable on its own regardless of your desire to sift through documents with an AI agent, as it enables all sorts of deeper explorations of large corpora.

**`bartleby write`** is the research agent: an interactive Q&A loop where you ask questions about your corpus and the agent searches, reads, and synthesizes answers with citations. It works well with paid models like `gpt-5-nano` and `gpt-5-mini`, and also with open-weights models like `gpt-oss:20b`, `qwen3:8b`, and `qwen3:30b` via Ollama.

A couple things to be aware of:

- Token costs can add up, especially during document summarization in `read` and during research sessions in `write`. You have knobs for this (e.g., how many pages to summarize per PDF, which can be zero). **The costs the tool shows are estimates.**
- I'm using the excellent (but pre-v0) [`sqlite-vec`](https://github.com/asg017/sqlite-vec) plugin for SQLite. There might be some instability there.

---

## Installation

### Prerequisites

Install system dependencies:

```bash
brew install tesseract
brew install uv
```

### Install Bartleby

From the project directory:

```bash
uv tool install .
```

This installs `bartleby` as a command-line tool in an isolated environment.

For development:

```bash
uv tool install --editable .
```

### Install Playwright browsers (optional, for HTML support)

If you want to process HTML files, install the Chromium browser for Playwright:

```bash
uv run playwright install chromium
```

This only needs to be done once. Skip this if you only process PDFs.

---

## Quick start

### 1. Configure

Run the setup wizard to choose your LLM provider, model, and other settings:

```bash
bartleby ready
```

This walks you through configuring worker threads, LLM provider/model, API keys, summarization depth, and temperature. Settings are saved to `~/.bartleby/config.yaml`.

### 2. Create a project

```bash
bartleby project create my-research
```

This creates a project directory and sets it as your active project. All subsequent commands use the active project by default.

### 3. Process documents

```bash
bartleby read --files /path/to/your/pdfs
```

Point this at a directory of PDFs (or HTML files) and Bartleby will extract text, generate embeddings, and optionally create LLM-powered summaries. Everything goes into a SQLite database in your project.

### 4. Ask questions

```bash
bartleby write
```

This starts an interactive research session. Ask questions about your corpus and the agent will search, read, and synthesize answers with source citations:

```
>: What does this corpus have to say about PM2.5 and equity?
  ✓ Listed documents (3 documents) ................... 0.2s
  ✓ Read summary (WANG-ET-AL_2024.pdf) ............... 0.3s
  ✓ Searched text (2 results) ........................ 3.1s
  ✓ Read passage (7 chunks) .......................... 6.0s
⠇ Thinking...

[Markdown-formatted answer with citations]

↑23.6k/↓5.4k/+29.0k (~$0.00)
```

Type `/save` to save the last answer as a timestamped report. Press `Ctrl+C` to exit.

---

## Command reference

### `bartleby ready`

Interactive configuration wizard. Asks for:

| Setting | Default | Description |
|---------|---------|-------------|
| Worker threads | 4 | Parallel processing threads for `read` |
| LLM provider | anthropic | `anthropic`, `openai`, or `ollama` |
| Model | varies by provider | Model name (e.g., `claude-3-5-sonnet-20241022`) |
| API key | — | Required for Anthropic/OpenAI; can also use env vars |
| Pages to summarize | 10 | Per-PDF page limit for summarization (0 = skip) |
| Temperature | 0 | 0 = deterministic, 1 = creative |

**API keys** can be provided in the config or via environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. For Ollama, configure the server URL (default `http://localhost:11434`) or set `OLLAMA_API_BASE`.

Config is saved to `~/.bartleby/config.yaml`.

### `bartleby project`

Manage project workspaces. Each project gets its own database, document archive, and output directory.

```
bartleby project create <name>    # Create and activate a new project
bartleby project list             # List all projects
bartleby project use <name>       # Switch active project
bartleby project info [name]      # Show project details (defaults to active)
bartleby project delete <name>    # Delete a project and its data (-y to skip prompt)
```

**Project directory structure:**

```
~/.bartleby/projects/<name>/
├── bartleby.db       # SQLite database (text, embeddings, summaries)
├── archive/          # Original PDF files (deduplicated by content hash)
└── book/             # Output artifacts
    ├── findings/     # Auto-saved Q&A results and research notes
    ├── report-*.md   # Saved reports (via /save)
    └── log.json      # Session log with tool calls and token usage
```

### `bartleby read`

Process PDF and HTML documents into the project database.

```bash
bartleby read --files <path> [options]
```

| Option | Description |
|--------|-------------|
| `--files <path>` | Path to a file or directory of PDFs/HTML (required) |
| `--project <name>` | Target project (defaults to active) |
| `--max-workers <n>` | Worker threads (default: from config) |
| `--model <name>` | Override LLM model for summarization |
| `--provider <name>` | Override LLM provider (`anthropic` or `openai`) |
| `--verbose` | Show debug output |

**Processing pipeline:**

1. Converts HTML to PDF (if applicable) via Playwright/Chromium
2. Extracts text from PDFs using PyMuPDF
3. Falls back to OCR (Tesseract) for image-based pages
4. Chunks text into segments (~400 characters with overlap)
5. Generates vector embeddings (BAAI/bge-base-en-v1.5)
6. Creates LLM-powered summaries for the first N pages (if configured)
7. Stores everything in SQLite with full-text search (FTS5) and vector search (sqlite-vec)

Supported file types: `.pdf`, `.html`, `.htm`

### `bartleby write`

Interactive research agent for investigating your document corpus.

```bash
bartleby write [options]
```

| Option | Description |
|--------|-------------|
| `--project <name>` | Target project (defaults to active) |
| `--verbose` | Show debug output and full tracebacks |

**In-session commands:**

| Command | Description |
|---------|-------------|
| `/save` | Save the last answer as `book/report-YYYYMMDDHHmm.md` |
| `Ctrl+C` | Exit the session |

The agent has access to search tools (keyword and semantic), document reading tools, summarization, and note-taking. Each question-answer pair is auto-saved to `book/findings/` for continuity across the session. Token usage and estimated costs are displayed after each answer.

---

## Supported LLM providers

| Provider | Default model | Vision support | Notes |
|----------|--------------|----------------|-------|
| Anthropic | `claude-3-5-sonnet-20241022` | Claude 3+ models | Requires API key |
| OpenAI | `gpt-4-turbo` | GPT-4 vision models | Requires API key |
| Ollama | `llama3.2` | No | Requires local server |

Vision-capable models can use page images during summarization for better results. Non-vision models fall back to text-only.
