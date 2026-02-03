# Bartleby, the Scrivener

A powerful document processing tool that extracts text from PDFs and HTML files, generates embeddings, and optionally creates LLM-powered summaries.

---

## Background

I have found it useful to let an AI agent (e.g. Claude Code) run wild in a SQLite database containing the extracted text from a bunch of documents. I've explored giving that agent various tools to explore the SQLite database more effectively, including enabling full-text and semantic searching. This provides a toolkit _and_ agent to generate reports based on caches of PDF documents.

### The parser — `bartleby read`.

OCR-ing and parsing PDFs (and converting HTML files) into a SQLite database, then paginating, summarizing, chunking, and embedding: These are valueable tasks regardless of your desire to sift through them with an AI agent. In fact, that workflow is something I use frequently, as it enables all sorts of deeper explorations of large corpora. So, I made this a standalone command.

HTML files are automatically converted to PDF using Playwright/Chromium before processing, allowing you to process web pages alongside traditional PDFs.

Some gotchas with this: If you've set up an LLM to summarize pages for you, it can burn through tokens pretty fast on summarization, but you have a knob: how many pages of each PDF to summarize. You can tell it to only do the first n pages, where n can be zero.

Also, I'm using the excellent (but pre-v0) `sqlite-vec` plugin for SQLite, [here](https://github.com/asg017/sqlite-vec). There might be some instability there.

### The writer — `bartleby write`.

This is an experiment: Can an agent run RAG for you on a prepared corpus? Can I write one that does? The answer to this is complicated. I have tested it, and it works reasonably well with paid models, such as `gpt-5-nano` and `gpt-5-mini` (though it can use ~100,000 tokens or so, so be careful). It also now works relaibly with `gpt-oss:20b`, `qwen3:8b`, and `qwen3:30b`--all open-weights models used with Ollama.

Be careful about token costs when using paid models. `gpt-5-nano` produces reports for pennies. `gpt-5-pro` or whatever might cost a good bit more. **The costs the tool shows are estimates!**

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

This will install `bartleby` as a command-line tool in an isolated environment.

In dev, you might want to run:

```bash
uv tool install --editable .
```

### Install Playwright Browsers (for HTML support)

If you want to process HTML files, install the Chromium browser for Playwright:

```bash
uv run playwright install chromium
```

This only needs to be done once. You can skip this if you only process PDFs.

---

## Usage

### Quick Start

1. **Configure once** (saves settings to `~/.bartleby/config.yaml`):

```bash
bartleby ready
```

This inits your `bartleby` instance, asking for everything it needs.

2. **Run anywhere**:

```bash
bartleby read --files path/to/files --db path/to/db
```

Process both PDFs and HTML files from a directory, or specify a single file.

### Options

**`bartleby ready`** - Interactive configuration wizard

**`bartleby read`** - Process PDFs and HTML files
- `--files` (required): Path to a file or directory containing PDF/HTML files (`.pdf`, `.html`, `.htm`)
  - Note: `--pdfs` still works for backward compatibility
- `--db` (required): Path to database directory (created automatically if it doesn't exist)

**`bartleby write`** - Write a report (includes live Q&A `/search` mode after the report so you can interrogate sources)
- `--db` (required): Path to a database directory you've created with `bartleby read`.

---

## What `read` does

1. **Converts HTML to PDF** (if HTML files are provided) using Playwright/Chromium with Letter size (8.5×11")
2. **Extracts text** from PDFs using PyMuPDF
3. **OCR fallback** for image-based pages using Tesseract
4. **Chunks text** intelligently using LangChain text splitters
5. **Generates embeddings** using sentence-transformers (BAAI/bge-base-en-v1.5)
6. **Creates summaries** (optional) for the first N pages using vision-capable LLMs
7. **Stores everything** in SQLite with full-text search (FTS5) and vector search (vec0)

## What `write` does

- Generates a report on the given SQLite database, derived from your documents.

---

## A note on ~ vibe coding ~.

Yes, I vibe coded a lot of this codebase, though I've made some efforts to clean it up. Sorry.

---

## Other learnings

The open-weights models runnable on my computer have different reqs! `gpt-oss:20b`, for example, wants simpler, post-processed schemas. While paid models want Pydantic modelling everywhere.
