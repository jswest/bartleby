# Bartleby v1 migration spec

This document describes the work required to migrate Bartleby from its current shape (a single CLI with `read` and `write` subcommands) to its v1 shape (a CLI focused on ingestion plus a portable skill for agent harnesses). Read the root `README.md` and `skill/README.md` before reading this — they describe the destination state. This document describes the route.

The migration is destructive. Existing databases will not work after v1. Existing users will be told to re-ingest. Do not write migration code, schema upgraders, or compatibility shims.

---

## 1. Goals and non-goals

### Goals

- Split Bartleby into a CLI (`bartleby`, this repo) and a skill (`skill/`, a folder in this repo).
- Make the SQLite database the contract between the two pieces. The CLI writes it; the skill reads and writes findings back to it.
- Replace `bartleby write` with a set of small Python scripts the skill exposes to its agent.
- Replace `bartleby book` with a single `bartleby logs` command.
- Add first-class sessions, with script-level enforcement of memory-off behavior.
- Make summaries whole-document and one-shot. Drop the first-N-pages approach.
- Make findings durable, chunked, and embedded into the same vector space as documents.
- Ship a polymorphic chunks table with `CHECK`-enforced source-kind discipline and typed Python insert helpers.

### Non-goals (v1)

- Map-reduce summarization. Single-shot only. Defer to v2.
- Cross-session memory beyond "search past findings" — no automatic injection, no "remember this for me," no summarization of past sessions.
- Per-project config beyond what's in `~/.bartleby/config.yaml`.
- Schema migration tooling. Break and re-ingest.
- A rich `bartleby book` UI. Just `bartleby logs`.
- An MCP server. The skill is plain scripts.

---

## 2. Architecture

```
                       ┌──────────────────┐
                       │  bartleby ready  │
                       │  bartleby scribe │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  SQLite database │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  bartleby skill  │
                       └──────────────────┘
```

### Repository layout (target)

```
.
├── README.md                  # the root README, already drafted
├── pyproject.toml
├── uv.lock
├── bartleby/                  # the CLI package
│   ├── __init__.py
│   ├── cli.py                 # entry point, dispatches subcommands
│   ├── config.py              # ~/.bartleby/config.yaml read/write
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.py          # DDL, schema version constant
│   │   ├── connection.py      # open/init helpers, sqlite-vec loading
│   │   └── chunks.py          # typed insert helpers (see §4.3)
│   ├── ingest/                # everything `bartleby scribe` needs
│   │   ├── __init__.py
│   │   ├── docling.py         # the one converter — handles pdf/html/md
│   │   ├── text.py            # plain-text fallback for .txt
│   │   ├── chunk.py
│   │   ├── embed.py
│   │   └── summarize.py       # single-shot whole-document summarizer (structured output)
│   ├── commands/              # one file per subcommand
│   │   ├── ready.py
│   │   ├── project.py
│   │   ├── scribe.py
│   │   ├── session.py
│   │   ├── embed.py
│   │   └── logs.py
│   └── providers/             # LLM provider clients (anthropic, openai, ollama)
└── skill/
    ├── README.md              # the skill README, already drafted
    ├── SKILL.md               # opinionated research methodology
    └── scripts/
        ├── list_documents.py
        ├── search.py
        ├── read_chunks.py
        ├── read_document.py
        ├── save_summary.py
        └── save_finding.py
```

The skill's scripts import from `bartleby.db` and `bartleby.providers` — there is one Python package, two surfaces. The skill folder is portable: copying `skill/` to `~/.claude/skills/bartleby/` works because the scripts shell out to the `bartleby` CLI for anything that needs config (notably query embedding).

### What the CLI owns

- Configuration (`bartleby ready`).
- Projects (`bartleby project ...`).
- Ingestion (`bartleby scribe`).
- Sessions (`bartleby session ...`).
- The embedding endpoint used by the skill (`bartleby embed`).
- The audit-log viewer (`bartleby logs`).
- The shared Python library that owns the schema, the connection helpers, and the typed insert functions.

### What the skill owns

- A `SKILL.md` that codifies research methodology.
- The six scripts the agent calls.
- Nothing that writes to the chunks table directly. All writes go through `bartleby.db.chunks` helpers.

---

## 3. Database schema

This is the canonical schema for v1. Implement it in `bartleby/db/schema.py` as a single DDL string plus a `SCHEMA_VERSION` constant.

`SCHEMA_VERSION = 2`

### Tables

```sql
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Required keys: schema_version, embedding_model, embedding_dim,
-- sqlite_vec_version, bartleby_version, created_at.

CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY,
    file_hash TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,         -- path in the project's archive/, original extension preserved
    page_count INTEGER,              -- NULL for .md / .txt sources
    token_count INTEGER,             -- approximate (tiktoken cl100k_base), computed at ingest
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(document_id) ON DELETE CASCADE,
    title TEXT NOT NULL,             -- short human title, surfaces in list_documents
    description TEXT NOT NULL,       -- one-line hook, surfaces in list_documents
    text TEXT NOT NULL,
    model TEXT NOT NULL,             -- which model wrote it
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- Unique on document_id: at most one summary per document.

CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,       -- memorable name, e.g. "mighty-grove"
    memory_enabled INTEGER NOT NULL DEFAULT 1,  -- 0 = no findings in search
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT                    -- optional, cosmetic only
);

CREATE TABLE findings (
    finding_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,       -- one-line hook for browsing prior findings
    body TEXT NOT NULL,              -- markdown
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE finding_citations (
    finding_id INTEGER NOT NULL REFERENCES findings(finding_id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    PRIMARY KEY (finding_id, chunk_id)
);

CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('document', 'summary', 'finding')),
    source_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    section_heading TEXT,            -- nullable, set when Docling provides it
    content_type TEXT,               -- nullable, set when Docling provides it
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_kind, source_id, chunk_index)
);

CREATE INDEX idx_chunks_source ON chunks(source_kind, source_id);

CREATE TABLE audit_logs (
    audit_log_id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT,                  -- JSON-encoded arguments
    result_summary TEXT,             -- short, agent-facing summary; nullable
    duration_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_session ON audit_logs(session_id, created_at);
```

### Virtual tables

```sql
-- FTS5 over all chunks
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    section_heading,
    content='chunks',
    content_rowid='chunk_id'
);

-- sqlite-vec embedding store
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[768]              -- BAAI/bge-base-en-v1.5 = 768 dims
);
```

Keep `chunks_fts` and `chunks_vec` in sync with `chunks` via the typed insert helpers in §4.3 — not via triggers. Triggers across virtual tables get finicky and we'd rather have one chokepoint in Python.

### Polymorphic FK note

`chunks.source_id` is not a foreign key to any single table — it references `documents.id`, `summaries.id`, or `findings.id` depending on `source_kind`. SQLite can't enforce this. The `CHECK` constraint on `source_kind` prevents typos. The typed insert helpers in §4.3 prevent mismatches. Don't add anything else; we've decided to live with the polymorphic-FK smell.

---

## 4. Shared Python library

### 4.1 `bartleby/db/connection.py`

A single function `open_db(project_name: str | None = None) -> sqlite3.Connection` that:

1. Resolves the project (active project if `None`).
2. Opens the SQLite file at `~/.bartleby/projects/<name>/bartleby.db`.
3. Loads `sqlite-vec` via `conn.enable_load_extension(True); sqlite_vec.load(conn)`.
4. Sets `PRAGMA foreign_keys = ON;`.
5. Verifies `meta.schema_version` matches `SCHEMA_VERSION`. Raises a clear error if not.
6. Returns the connection.

Also expose `init_db(project_name: str) -> None` which runs the DDL and populates the `meta` table. Called by `bartleby project create`.

### 4.2 `bartleby/db/schema.py`

Holds `SCHEMA_VERSION = 2` and `DDL: str` (the full schema from §3 as a single string). Also a `META_KEYS` list of required meta keys for sanity checking.

### 4.3 `bartleby/db/chunks.py`

The chokepoint for all writes to the chunks table. Three functions:

```python
def insert_document_chunks(
    conn: sqlite3.Connection,
    document_id: int,
    chunks: list[ChunkInput],
) -> list[int]: ...

def insert_summary_chunks(
    conn: sqlite3.Connection,
    summary_id: int,
    chunks: list[ChunkInput],
) -> list[int]: ...

def insert_finding_chunks(
    conn: sqlite3.Connection,
    finding_id: int,
    chunks: list[ChunkInput],
) -> list[int]: ...
```

Where `ChunkInput` is a dataclass:

```python
@dataclass
class ChunkInput:
    text: str
    embedding: list[float]            # 768-dim
    chunk_index: int
    section_heading: str | None = None
    content_type: str | None = None
```

Each function:

1. Validates inputs (non-empty text, correct embedding dimension, monotonic chunk_index).
2. Inserts into `chunks` with the correct `source_kind` (hardcoded per function — that's the point).
3. Inserts into `chunks_fts` and `chunks_vec` in the same transaction.
4. Returns the list of inserted chunk IDs.

Also expose `delete_chunks_for(conn, source_kind, source_id)` for replacing summaries/findings on update — deletes from `chunks`, `chunks_fts`, and `chunks_vec` in one transaction.

**Nothing else may write to `chunks` directly.** Scripts that need to write chunks import from this module. Code review should reject direct INSERTs into `chunks` outside this file.

---

## 5. CLI commands

### 5.1 `bartleby ready`

Interactive wizard. Reads/writes `~/.bartleby/config.yaml`. Settings:

| Key | Default | Description |
| --- | --- | --- |
| `llm_provider` | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `llm_model` | provider-default (see below) | Model name |
| `api_key` | — | Stored or sourced from env |
| `summary_depth` | `one-shot` | `none` \| `one-shot` |
| `temperature` | 0 | 0..1 |
| `max_summarize_tokens` | 50000 | Truncate document input to first N tokens before summarizing. See §5.3. |
| `max_read_tokens` | 50000 | Threshold for `read_document --force` |
| `ollama_base_url` | `http://localhost:11434` | Only if provider is ollama |

Provider defaults for `llm_model`:

| Provider | Default |
| --- | --- |
| anthropic | `claude-haiku-4-5` |
| openai | `gpt-5-mini` |
| ollama | `gpt-oss:20b` |

The Ollama default is a guess at "runs on a reasonable laptop" — users with smaller hardware will need to override. The README calls this out.

Behavior: identical to the existing `ready` command except for:
- New `summary_depth` (replaces `pages_to_summarize`).
- New `max_read_tokens`.
- **Removed**: `max_workers`. Ingestion is sequential in v1 — see §5.3.

### 5.2 `bartleby project`

Subcommands: `create`, `list`, `use`, `info`, `delete`. Same as current behavior. The only change is that `create` initializes the new schema via `init_db` and populates the `meta` table.

`info` should print the schema version, embedding model, document count, chunk count by source kind, session count, and finding count.

### 5.3 `bartleby scribe`

```
bartleby scribe --files <path> [--project <name>] [--model <name>] [--provider <name>] [--verbose]
```

(Renamed from `bartleby read`. Read commands belong to the agent; the CLI scribes.)

Note the removed flags:
- `--max-workers` is gone. Ingestion is **sequential** in v1.
- `--docling` is gone. Docling is the only converter for PDF/HTML/MD (always on).

#### Supported file types

Docling is a hard dependency of `bartleby scribe`. There is one ingest path, not two.

| Extension | Converter | Chunker |
| --- | --- | --- |
| `.pdf` | Docling PDF backend (with OCR) | Docling `HybridChunker` |
| `.html` / `.htm` | Docling HTML backend | Docling `HybridChunker` |
| `.md` | Docling Markdown backend | Docling `HybridChunker` |
| `.txt` | UTF-8 read (Docling has no text reader) | Simple character chunker (~800 chars, ~100 overlap) |

Files are deduplicated by SHA-256 of content. Archived at `archive/<hash>/<hash>.<ext>` with the original extension preserved.

For `.md` and `.txt`: `page_count` is `NULL`. `token_count` is computed via `tiktoken.cl100k_base` over the full document text. Same encoder for all sources, regardless of provider.

Playwright and PyMuPDF are removed as dependencies. Tesseract is no longer invoked directly (Docling handles OCR internally for image-based PDFs).

#### Internal pipeline

1. Hash the source file and archive it at `archive/<hash>/<hash>.<ext>`.
2. Convert and chunk:
   - `.pdf` / `.html` / `.md`: pass to `DocumentConverter()`, then `HybridChunker(tokenizer=EMBEDDING_MODEL, max_tokens=400)`. The 400 cap leaves headroom for heading context against the embedder's 512 max sequence length.
   - `.txt`: read as UTF-8, run the simple character chunker. No headings.
3. Compute `token_count` over the full document text via `tiktoken.cl100k_base`. Insert the `documents` row. For Docling-converted documents, take `page_count` from the converter; for `.txt`, leave it `NULL`.
4. Embed chunks. Insert via `insert_document_chunks`.
5. If `summary_depth == 'one-shot'`, generate a whole-document summary (see §5.3.1 below). Insert into `summaries` with the model id. Chunk and embed the summary text with the same Docling chunker and insert via `insert_summary_chunks`.

Remove all code related to per-page summarization and the `pages_to_summarize` config key.

#### 5.3.1 Summarization contract

The summarizer is **structured-output only** across all providers. We enforce JSON to keep open-source models from drifting into "Here is your summary:" preambles, thinking tags, or stray markdown fences, and we ask for three fields in one call so we never pay for the same document three times:

```python
class DocumentSummary(BaseModel):
    title: str        # short human-readable title (≤ 60 chars, no filename, no quotes)
    description: str  # one-sentence hook (~20 words, ≤ 200 chars)
    text: str         # concise self-contained summary
```

The schema is rendered with field-level descriptions (via `Field(description=...)`) so providers that surface field descriptions in their structured-output mechanism see the guidance.

Per-provider invocation:

- **Anthropic**: tool-use with `DocumentSummary.model_json_schema()` as the tool's `input_schema`. Force the model to call that one tool. Parse `tool_use.input`.
- **OpenAI**: `chat.completions.parse(response_format=DocumentSummary)` — the SDK converts the Pydantic model to a strict JSON schema. Read `message.parsed`.
- **Ollama**: pass `format=DocumentSummary.model_json_schema()` to the chat call (Ollama supports JSON-schema-constrained output since v0.5). Parse the response content.

All three: validate the parsed JSON against `DocumentSummary` before persisting. If validation fails, raise and let the document fail ingest with a clear error (do not silently insert a malformed summary).

The `title` and `description` are stored on `summaries` and surfaced by `list_documents` so agents can triage the corpus without reading every summary `text`.

#### 5.3.2 Long-document truncation

If `len(tiktoken_encode(document_text)) > max_summarize_tokens`, truncate the input to the first `max_summarize_tokens` tokens before passing to the summarizer. After the LLM returns a valid `DocumentSummary`, append a deterministic note to the `text` field only (`title` and `description` are unaffected):

```
\n\n_Note: this summary is based on the first {N} tokens of a {M}-token document._
```

The note is appended in code, not via the prompt — we want the caveat to be guaranteed, not modeled.

This is independent of `max_read_tokens` (which gates `read_document --force`). They share a default of 50000 only because that's a reasonable shared ceiling, not because they're coupled.

### 5.4 `bartleby session`

Three subcommands:

```
bartleby session start [--no-memory]
bartleby session current
bartleby session end
```

- `start`: insert a row into `sessions` with a memorable name (use the existing word-pair generator from `bartleby write`). Set `memory_enabled = 0` if `--no-memory`. Print the session name and ID. Mark this session as active by writing it to `~/.bartleby/projects/<name>/.active_session`.
- `current`: print the active session for the active project (name, ID, memory state, created_at).
- `end`: set `ended_at = CURRENT_TIMESTAMP` on the active session and clear `.active_session`. This is cosmetic; nothing in the system requires sessions to be ended.

If a script (in the skill) needs the active session and `.active_session` is missing, it should auto-create one with default settings (`memory_enabled = 1`). Don't fail; sessions are bookkeeping, not a gate.

### 5.5 `bartleby embed`

```
bartleby embed <text>
```

Embeds `<text>` with the same model used at ingest (BAAI/bge-base-en-v1.5) and prints the resulting vector as JSON to stdout. Used by the skill's `search.py` for semantic queries.

If the model isn't loaded, this is the place to load it (lazy initialization). Each invocation pays the model load cost (~5–10s); we accept this for v1. No daemon, no caching.

The skill MUST invoke this command with list-form `subprocess.run(["bartleby", "embed", query], ...)` — never `shell=True`. With list-form, the query is `argv[1]` and the shell does not interpret it; no escaping or sanitization is required.

### 5.6 `bartleby logs`

```
bartleby logs [--session <name>] [--limit <n>]
```

Pretty-prints rows from `audit_logs`. Defaults: most recent session, limit 50. Columns: timestamp, tool name, args (truncated), duration. No fancy table library required; the existing `rich` usage is fine.

### 5.7 What gets removed

- `bartleby write` (entire command and its agent loop).
- `bartleby book` (entire command and all subcommands).
- All code under `bartleby/` related to the interactive write loop, findings-to-disk writing, log.json maintenance, and the `book/` directory output.
- Per-page summarization code in the ingest pipeline.
- The `book/` directory creation in `bartleby project create`.

Delete the code. Don't leave it behind under feature flags.

---

## 6. Skill scripts

All scripts live in `skill/scripts/` and follow these conventions:

- Arguments via `argparse`.
- JSON output to stdout. One JSON object per invocation. Schema documented in each script's docstring and in `SKILL.md`.
- Errors: print a JSON object `{"error": "<message>", "code": "<code>"}` to stdout and exit 1. Use stderr only for human-readable diagnostics that the agent shouldn't parse.
- All scripts accept `--project <name>` and fall back to the active project.
- All scripts that read or write findings or audit-log entries accept (and respect) the active session.

Each script must log its invocation to `audit_logs` (tool_name = script name, args_json = the parsed args, duration_ms, result_summary = a short string). The logging helper lives in `bartleby/db/audit.py` (create this module — add it to the layout in §2).

### 6.1 `list_documents.py`

```
list_documents [--project <name>] [--limit <n>] [--offset <n>]
```

Output:

```json
{
  "documents": [
    {
      "id": 12,
      "file_name": "WANG-ET-AL_2024.pdf",
      "title": "Detection gaps in the new PM2.5 NAAQS",
      "description": "Estimates how many people live in unmonitored hotspots under the tightened standard.",
      "page_count": 22,
      "token_count": 8430,
      "has_summary": true,
      "chunk_count": 47,
      "created_at": "2026-05-12T14:32:01"
    }
  ],
  "total": 138
}
```

`title` and `description` come from the document's `summaries` row and are `null` until a summary is written (either at ingest time or via `save_summary`). They are how agents triage the corpus without reading every summary `text`.

### 6.2 `search.py`

```
search "<query>" \
  [--documents] [--summaries] [--findings] \
  [--semantic] [--full-text] \
  [--in-documents <id,id,...>] \
  [--context <n>] [--limit <n>] [--project <name>]
```

Defaults:

- Source kinds: `--documents` only. `--summaries` and `--findings` must be explicit.
- Modes: `--semantic --full-text` (both on, results combined via RRF).
- Context: `--context 1` (one neighboring chunk on each side of each hit). Range: `0..5`.
- Limit: 20.

Behavior:

1. Resolve active session. If `--findings` is requested and `memory_enabled = 0`, silently drop the `--findings` flag and add a note to the response.
2. Resolve scope. If `--in-documents` is set: restrict `document` chunks to those `source_id`s; restrict `summary` chunks to summaries whose `document_id` is in the list; drop `finding` entirely (findings aren't tied to documents). Without `--in-documents`, every requested kind is unrestricted.
3. For full-text mode, query `chunks_fts` filtered by the scope.
4. For semantic mode, shell out to `bartleby embed "<query>"`, then query `chunks_vec` with the resulting vector, joined back to `chunks` and filtered by the scope.
5. If both modes are on, combine via Reciprocal Rank Fusion (RRF) with `k = 60`. Score is `sum(1 / (k + rank))` across the lists each result appears in. Sort descending.
6. Take top N hits. For each hit, fetch the surrounding chunks within the same `(source_kind, source_id)` whose `chunk_index` falls in `[hit_index - context, hit_index + context]`, excluding the hit itself. Clamp at source boundaries.
7. Compute `rank` (1-indexed position) and `normalized_score` (`score / top_score`) for each hit so agents have a triage signal that's readable on its own.

Output:

```json
{
  "query": "PM2.5 and equity",
  "modes": ["semantic", "full-text"],
  "source_kinds": ["document"],
  "memory_excluded": false,
  "in_documents": null,
  "context": 1,
  "results": [
    {
      "chunk_id": 4192,
      "source_kind": "document",
      "source_id": 12,
      "source_name": "WANG-ET-AL_2024.pdf",
      "chunk_index": 18,
      "section_heading": "Results: equity analysis",
      "text": "the matched chunk",
      "context_before": ["chunk 17 text"],
      "context_after":  ["chunk 19 text"],
      "rank": 1,
      "score": 0.0341,
      "normalized_score": 1.0
    }
  ]
}
```

Notes:

- `text` is always the hit alone. `context_before` and `context_after` are arrays in **document order** (ascending `chunk_index`), so concatenating `context_before + [text] + context_after` produces a contiguous passage. For `--context 2` at hit index 18: `context_before = [chunk 16 text, chunk 17 text]`, `context_after = [chunk 19 text, chunk 20 text]`. Arrays are empty when the hit sits at a source boundary.
- The agent **must cite the hit's `chunk_id`**, not anything drawn from the context arrays. The skill prompt (§7) makes this explicit.
- Context fetching is per-hit, never across different `(source_kind, source_id)` pairs. A hit at the start of a document does not pull in the end of the previous document.
- `--context 0` disables context entirely (returns the hit only, with empty arrays).
- `rank` is 1-indexed within the returned `results` list. `normalized_score` is `score / max(score)` (so the top hit is always `1.0`); it makes the relative strength of lower hits legible. Raw `score` values are tiny by design (RRF range ~`0.015–0.033`) and only comparable within a single query; SKILL.md tells agents to triage with `rank` first and use `normalized_score` to gauge spread.
- `in_documents` echoes the resolved `--in-documents` list (or `null` when unset) so the agent can confirm the scope it ran under.

If `memory_excluded` is true, the agent knows the user is in a no-memory session and findings are unreachable.

### 6.3 `read_chunks.py`

Two mutually exclusive modes (exactly one of `--document` or `--chunks` is required):

```
read_chunks --document <id> [--offset <n>] [--limit <n>] [--project <name>]
read_chunks --chunks <id,id,...> [--project <name>]
```

**Document mode** reads chunks from a single document, ordered by `chunk_index`. Default offset 0, default limit 50.

```json
{
  "mode": "document",
  "document": { "id": 12, "file_name": "WANG-ET-AL_2024.pdf" },
  "offset": 0,
  "limit": 50,
  "total": 47,
  "chunks": [
    {
      "chunk_id": 4175,
      "chunk_index": 0,
      "section_heading": "Abstract",
      "content_type": "text",
      "text": "..."
    }
  ]
}
```

**Chunks mode** looks up specific chunk_ids directly, regardless of source. Each returned chunk carries its `source_kind` / `source_id` / `source_name` so the agent can locate it. The response also lists which requested ids were not found.

```json
{
  "mode": "chunks",
  "requested": [4192, 4188, 9201],
  "missing": [],
  "chunks": [
    {
      "chunk_id": 4192,
      "source_kind": "document",
      "source_id": 12,
      "source_name": "WANG-ET-AL_2024.pdf",
      "chunk_index": 18,
      "section_heading": "Results: equity analysis",
      "content_type": "text",
      "text": "..."
    }
  ]
}
```

Chunks are returned in the order requested (de-duplicated). Chunks mode does not paginate — the caller is expected to ask for what they want.

### 6.4 `read_document.py`

```
read_document --document <id> [--summary | --full] [--force] [--project <name>]
```

- Default (no `--summary`/`--full`): return both summary and full text.
- `--summary`: summary only.
- `--full`: full text only.
- `--force`: bypass the `max_read_tokens` guard.

If `--full` (or default) would return more than `max_read_tokens` tokens and `--force` is not set, return an error:

```json
{
  "error": "Document exceeds max_read_tokens (50000). Pass --force to read anyway, or use read_chunks for paginated access.",
  "code": "DOCUMENT_TOO_LARGE",
  "token_count": 84200,
  "max_read_tokens": 50000
}
```

Successful output:

```json
{
  "document": { "id": 12, "file_name": "WANG-ET-AL_2024.pdf", "token_count": 8430 },
  "summary": "... (or null) ...",
  "full_text": "... (or null) ..."
}
```

### 6.5 `save_summary.py`

```
save_summary --document <id> --title <title> --description <desc> --text <text> [--project <name>]
```

Writes (or replaces) the agent-authored summary for a document. Inserts a row into `summaries` with `title`, `description`, and `text` (all required and non-empty), chunks and embeds the `text`, then inserts via `insert_summary_chunks`. If a summary already exists for that document, delete the old summary's chunks (via `delete_chunks_for`) and replace.

`title` and `description` are how the document shows up in `list_documents`, so the SKILL.md nudges agents to make them informative.

Note: the ingest pipeline also writes summaries. Agent-saved summaries and ingest-time summaries share the same table. They're both tagged `source_kind = 'summary'` in chunks. We accept this — if the agent saves a "better" summary, it overwrites. If you want to differentiate later (v2), add a `created_by` column.

Output:

```json
{
  "summary_id": 88,
  "document_id": 12,
  "chunk_ids": [9201, 9202, 9203]
}
```

### 6.6 `save_finding.py`

```
save_finding --title <title> --description <desc> --body-file <path> [--citations <chunk_id,chunk_id,...>] [--project <name>]
```

Why `--body-file` and not `--body`: findings are markdown, sometimes long, and shell-escaping them is a nightmare. Agents write the body to a tempfile and pass the path.

Inserts a row into `findings` tied to the active session, with `title`, `description`, and `body` (all required and non-empty). `description` is a one-line hook future agents see when triaging prior findings via `search --findings`. Chunks and embeds the body using Docling's `HybridChunker(tokenizer=EMBEDDING_MODEL, max_tokens=400)` — the same chunker used by `bartleby scribe`. Docling parses the finding's markdown structurally, so the resulting chunks carry `section_heading` and `content_type` metadata just like ingested documents.

Inserts via `insert_finding_chunks`. Inserts citation rows into `finding_citations` for each `chunk_id` in `--citations`.

Output:

```json
{
  "finding_id": 17,
  "session_id": 4,
  "session_name": "mighty-grove",
  "chunk_ids": [11023, 11024],
  "citation_count": 6
}
```

---

## 7. `SKILL.md`

For v1, port the existing `bartleby write` system prompt into `skill/SKILL.md` with minimal changes. The user (jswest) will rewrite it later into something deeply opinionated and conservative.

The v1 port should at minimum cover:

- A one-paragraph identity statement (who the agent is, what corpus it's working with).
- A description of each script and when to call it.
- The default behavior: search before reading, prefer summaries before full documents, cite chunks.
- **The citation rule for `search` context**: each result has `text` (the hit) plus `context_before` and `context_after`. Cite the hit's `chunk_id`. The context arrays are for reading comprehension only — never derive a citation from them, and never quote them as if they were the hit.
- The memory contract: findings are hints, not evidence. Never cite a finding. If the user asks to ignore prior memory, stop and instruct them to restart with `bartleby session start --no-memory`.
- The output format: markdown responses with inline citations. Use `save_finding` to persist anything worth keeping.

Keep it short. The rewrite is coming.

---

## 8. Order of operations

Do these in order. Each step should be a single commit (or small commit series) with a working build.

1. **Repo restructure.** Move existing code into the layout in §2. Create empty `skill/` directory. Update `pyproject.toml` if entry points change.
2. **New schema and connection layer.** Implement `bartleby/db/schema.py`, `bartleby/db/connection.py`, `bartleby/db/chunks.py`, `bartleby/db/audit.py`. Write unit tests that hit a temporary SQLite file: insert documents/summaries/findings, verify FTS and vec indexes stay in sync, verify the CHECK constraint rejects bad source_kinds.
3. **Update `bartleby project create`** to use the new schema. Update `bartleby project info` to show the new stats.
4. **Update `bartleby ready`** to write the new config keys. Remove `pages_to_summarize` and `max_workers` from the config schema and from any code that reads them.
5. **Rename `bartleby read` → `bartleby scribe`** and rebuild the ingest pipeline. Make it sequential (delete `ProcessPoolExecutor` and `_process_pdf_worker`). Drop the `--docling` flag — Docling is always on. Drop Playwright (and PyMuPDF, and direct Tesseract) — Docling handles PDF/HTML/MD with internal OCR. Add `.md` and `.txt` support (`.txt` uses a simple character chunker since Docling has no text reader). Replace per-page summarization with a structured-output one-shot summarizer (Pydantic `DocumentSummary` enforced for anthropic/openai/ollama; see §5.3.1). Apply `max_summarize_tokens` truncation with a deterministic appended note (§5.3.2). Compute `documents.token_count` via tiktoken `cl100k_base`. Route all chunk writes through `insert_document_chunks` and `insert_summary_chunks`. End-to-end smoke test: ingest one PDF, one HTML, one MD, one TXT; verify `documents`/`summaries`/`chunks`/`chunks_fts`/`chunks_vec` are populated correctly, summaries are valid JSON, and a >50k-token document gets the truncation note.
6. **Add `bartleby session`** subcommand. Active-session file at `~/.bartleby/projects/<name>/.active_session`.
7. **Add `bartleby embed`** subcommand.
8. **Add `bartleby logs`** subcommand.
9. **Delete `bartleby write`** entirely. Delete `bartleby book` entirely. Delete the `book/` directory creation. Delete per-page summarization code. Delete the cross-encoder reranker (`RERANKER_MODEL` and all uses). Delete all dead imports.
10. **Implement the six skill scripts** in `skill/scripts/`, in this order: `list_documents`, `read_chunks`, `read_document`, `search`, `save_summary`, `save_finding`. Each script gets a smoke test that exercises its happy path against a tiny seeded database.
11. **Port the existing `write` system prompt** into `skill/SKILL.md`. Adjust references to the new script set.
12. **Move `SKILL-README.md` into `skill/README.md`.** Cross-check that everything described in the READMEs actually works.
13. **Final pass.** Run `bartleby ready`, create a project, ingest a small corpus (PDF + MD + TXT), start a session, exercise each script manually from the shell. Verify `bartleby logs` shows the calls.

Stop after each step and confirm the build is green before moving to the next.

---

## 9. Testing

Per-step smoke tests are required (see §8). Beyond that:

- **Schema tests.** Open a fresh DB, run DDL, verify all tables and virtual tables exist, verify `meta` is populated, verify CHECK constraints reject bad inputs.
- **Chunk helpers tests.** Insert via each helper, verify chunks/FTS/vec are in sync, verify `delete_chunks_for` cleans all three.
- **RRF unit test.** Given two ranked lists of chunk IDs with known overlap, verify the fused order matches hand-computed RRF scores.
- **Memory-off enforcement test.** Create a session with `memory_enabled = 0`, save a finding, run `search` with `--findings`, assert findings are excluded and `memory_excluded: true` is in the response.
- **Document-too-large test.** Ingest a small document, set `max_read_tokens` to 1, call `read_document` without `--force`, assert the error code.

No end-to-end LLM tests. Don't burn tokens in CI.

---

## 10. Things to ask about, not guess at

If you hit any of these, stop and ask:

- The shape of any JSON output not specified above.
- Any change to the public CLI surface (flag names, command names, argument order).
- Any deviation from the schema in §3, including added columns, indexes, or virtual tables.
- Any change to the order of operations in §8 that would leave the build red between commits.
- Whether to preserve behavior you find in the current codebase that isn't described in this spec. (Default answer: no.)

---

## 11. Resolved questions (decision log)

Decisions taken during spec review, recorded here so future-you doesn't have to re-derive them.

- **CLI rename**: `bartleby read` → `bartleby scribe`. "Read" belongs to the agent; the CLI scribes. Internal package paths follow (`commands/scribe.py`).
- **Text-only files (`.md`, `.txt`)**: supported in `bartleby scribe`. `page_count` is `NULL`. Dedupe by SHA-256 with original extension preserved in the archive. See §5.3.
- **Parallel ingestion**: removed entirely. Pipeline is sequential. `max_workers` config key and `--max-workers` flag are deleted.
- **Docling is required.** It is the only converter for `.pdf`/`.html`/`.md`. The `--docling` flag is removed. For `.txt`, a simple character chunker is used (Docling has no text reader).
- **Playwright, PyMuPDF, and direct Tesseract are removed.** Docling handles PDF/HTML conversion and image-PDF OCR internally. One less Chromium download in install.
- **Summarizer structured output**: Pydantic `DocumentSummary(text: str)` enforced across all three providers (Anthropic tool-use, OpenAI `response_format`, Ollama `format=` JSON schema). Validation failures raise. See §5.3.1.
- **Long-document summarization**: configurable via `max_summarize_tokens` (default 50000). If input exceeds, truncate to first N tokens (tiktoken `cl100k_base`) before the LLM call. Append a deterministic note to the saved summary. See §5.3.2.
- **Sessions**: explicit `bartleby session start` is only required for `--no-memory`. Otherwise the skill auto-creates a default session on first script call. Both READMEs explain this.
- **Query embedding**: skill shells out to `bartleby embed` via list-form `subprocess.run`. No shell escaping. No sanitization. No daemon.
- **Default LLM models**: anthropic = `claude-haiku-4-5`, openai = `gpt-5-mini`, ollama = `gpt-oss:20b`. The Ollama default assumes "reasonable laptop" hardware; the README notes the override.
- **Embedding and reranker models**: keep `BAAI/bge-base-en-v1.5` (768 dims, matches schema). Drop the cross-encoder reranker entirely — RRF replaces it.
- **Foreign keys**: reference named PK columns (`REFERENCES documents(document_id)`, etc.) so `USING (document_id)` joins work cleanly.
- **Table naming**: plural (`audit_logs`). The CLI subcommand `bartleby logs` reads from this table.
- **`chunks_fts.content_rowid`**: `'chunk_id'`.
- **Finding chunking**: uses Docling's `HybridChunker` with `max_tokens=400` (headroom against the 512 embedder limit). Same chunker as `bartleby scribe`.
- **Search context window**: `search` results auto-include neighboring chunks (`--context`, default 1, range 0..5) under `context_before` / `context_after` arrays. Context never crosses `(source_kind, source_id)` boundaries. Agents must cite the hit's `chunk_id`, not anything from the context arrays. `read_chunks` is unchanged — its own offset/limit pagination already provides surrounding text.
- **`documents.token_count`**: computed via `tiktoken.cl100k_base`. Approximate across providers; that's acceptable for a `--force` gate. README flags this.
- **Schema v2 — title/description on summaries and findings**: `summaries` gains `title` and `description` (both `NOT NULL`); `findings` gains `description` (`NOT NULL`). `SCHEMA_VERSION` bumped to 2. Motivation: an agent using the skill reported it could only triage the corpus by filename + raw summary text, and that prior findings were similarly opaque. Surfacing a short title and one-line hook in `list_documents` (and in finding lookups) makes the corpus browsable. Per project policy: no migration code, users re-ingest. The summarizer now returns all three fields in one structured-output call, so we don't pay for the document text three times.
- **`search --in-documents <id,id,...>`**: scopes a search to the listed documents' chunks (and to their summaries' chunks). `finding` source-kind is dropped when this flag is set because findings aren't tied to documents. Resolved scope echoes back in the response under `in_documents`. Motivation: agents identified candidate documents from `list_documents`/`search` and wanted to drill in without re-reading them whole.
- **`search` result fields — `rank` and `normalized_score`**: each result now carries a 1-indexed `rank` plus `normalized_score = score / max(score)`. Raw RRF `score` is preserved but tiny by design (~`0.015–0.033`) and only comparable within one query. SKILL.md tells agents to triage with `rank` first and use `normalized_score` to gauge spread.
- **`read_chunks --chunks <id,id,...>`**: a second mode for the same script (mutually exclusive with `--document`) that fetches arbitrary chunks by `chunk_id` regardless of source. Returns each chunk with its `source_kind` / `source_id` / `source_name` plus a `requested` and `missing` list. Useful for revisiting cited chunks without re-running a search.
- **No running citation tracker**: the user-facing agent feedback proposed a `bartleby skill cite <chunk_id>` command + a `--use-tracked-citations` flag on `save_finding`. We pushed back: `save_finding` already provides a durable place to stash chunk_ids the agent doesn't want to lose, and adding a stateful buffer per session introduces a parallel mental model with its own edge cases (when does it clear? what if the session ends mid-research?). Instead, the SKILL.md instructs agents to write interim findings (one-line body is fine) when the context grows long.