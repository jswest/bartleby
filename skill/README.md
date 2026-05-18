# The Bartleby Skill

A skill that lets an agent romp through a Bartleby corpus--searching, reading, synthesizing, citing, and saving findings--inside any compliant harness (Claude Code, Cowork, or similar).

The skill is BYO-model. It works against whatever model your harness runs.

---

## What this is, and what it isn't

**It is:** a set of small Python scripts that talk to a Bartleby SQLite database, plus a `SKILL.md` that tells the agent how to use them well. The skill is opinionated — it has views about what counts as evidence, when to search vs. read, and how to behave when memory is on or off.

**It isn't:** a way to ingest documents. That's the [`bartleby` CLI](../README.md). The skill assumes the database already exists and the corpus is already chunked, embedded, and indexed.

---

## Prerequisites

1. The `bartleby` CLI is installed and on your `PATH`. The skill shells out to it for embedding queries (semantic search). Everything else (config, schema, project resolution) is imported from the installed `bartleby` package directly. (See the [main README](../README.md) for install instructions.)
2. A Bartleby project exists, with documents already ingested (`bartleby scribe`).
3. The project is the active project, *or* each script is invoked with `--project <name>`.

Every script opens the project DB via the shared runner, which validates the schema version on the spot and refuses to run against an incompatible database. If a session isn't active, the runner auto-creates one with memory on (see "How sessions work" below).

---

## Installation

Copy the skill directory into your harness's skills location.

**Claude Code:**

```
cp -r skill ~/.claude/skills/bartleby
```

**Other harnesses:** consult the harness's docs for where skills live. The skill is a self-contained folder; you should be able to drop it anywhere a compliant harness reads skills from.

---

## What the skill exposes

The skill ships a `scripts/` directory containing small Python scripts that the agent calls. Each takes arguments, prints JSON to stdout, and exits non-zero on error.

| Script | Purpose |
| --- | --- |
| `list_documents` | Enumerate documents in the corpus (file names, IDs, page/token/chunk counts, summary status). |
| `search` | Unified search across documents, summaries, and findings. Supports keyword (FTS5), semantic (vector), and hybrid (RRF) modes. Each hit ships with neighboring chunks for reading context (configurable via `--context`). |
| `read_chunks` | Read a window of chunks from a document. Paginated via `--offset` and `--limit`. |
| `read_document` | Read a full document and/or its summary. Refuses oversized documents without `--force`. |
| `save_summary` | Save an agent-authored summary back into the database (chunked and embedded). |
| `save_finding` | Save a finding (markdown text + structural citations) into the database. |

The scripts wrap a shared Python library that owns all writes to the chunks table. Source-kind discipline (`document` vs. `summary` vs. `finding`) is enforced both by a `CHECK` constraint at the SQL layer and by typed insert helpers in the library, so agents can't accidentally mislabel chunks.

For full argument-level contracts, see [`SKILL.md`](./SKILL.md) and the script docstrings.

---

## How sessions work

Every agent run happens inside a *session*. Sessions are rows in the database with an ID and a memorable name (e.g., `mighty-grove`). Findings and audit log entries are tagged with a `session_id`.

Sessions don't really "end" — there's no end-state to enforce. They're just a way to group related work and to thread provenance through the database.

**Starting a session:**

```
bartleby session start
```

This prints the session ID and name. The skill picks it up automatically via the active session.

**You usually don't need to.** If no session is active when the skill's first script runs, the skill auto-creates one with default settings (memory on). Run `bartleby session start` explicitly only when you want `--no-memory`.

**Memory:**

By default, the `search` script can return findings from any prior session. This lets the agent build on past research without re-deriving conclusions.

If a user wants the agent to ignore prior findings, they start a memory-off session:

```
bartleby session start --no-memory
```

In a memory-off session, the `search` script silently excludes findings from results, *regardless of what flags the agent passes*. This is enforced at the script level, not via prompt. The agent literally cannot reach prior findings.

If a user asks the agent mid-session to "ignore previous memory" or similar, the skill instructs the agent to stop and tell the user to restart with `bartleby session start --no-memory`. The skill does not attempt to honor memory-off requests within an already-running session.

---

## How findings work

Findings are the durable output of a research session. They live in the `findings` table as markdown text, *plus* a `finding_citations` join table linking each finding to the source chunks it rests on.

Findings are chunked and embedded into the same vector space as documents and agent-generated summaries. This means cross-session memory is just semantic search — the agent searches its own past findings the same way it searches the corpus.

Findings are tagged with `source_kind = 'finding'` and excluded from search by default. The agent must opt in via `--findings` to include them. The skill's prompt guides the agent on when to do this (typically: at the start of a new topic, to check for prior relevant work; never as primary evidence in a citation).

---

## What's stored, and where

Everything queryable lives in the project's `bartleby.db`:

- `documents` — one row per ingested file
- `chunks` — polymorphic: documents, summaries, and findings all chunk into here, tagged with `source_kind` and `source_id`
- `summaries` — one row per document (single-shot, whole-document)
- `sessions` — one row per agent session
- `findings` — one row per saved finding, with markdown text
- `finding_citations` — join table from findings to the chunks they cite
- `audit_logs` — every tool call the agent made, append-only, never read by the agent

No sidecar files. If you want to export a finding to disk, ask the agent to write it.

---

## A note on the skill's opinions

`SKILL.md` is deliberately opinionated and conservative (small-c). It defaults the agent toward:

- Searching before reading
- Reading summaries before reading full documents
- Citing source chunks, not paraphrasing without provenance
- Treating prior findings as hints, never as citable evidence
- Stopping and asking when the user's intent is ambiguous

If you want different defaults, edit `SKILL.md`. It's plain prose. That's the point.

---

## Troubleshooting

**"Schema version mismatch."** The database was created by a different version of the `bartleby` CLI. Either upgrade the CLI or re-ingest the corpus.

**"No active project."** Run `bartleby project use <name>` or pass `--project` via your harness's environment.

**Agent is calling tools but getting empty results.** Check `bartleby logs` for what it actually queried. The most common cause is searching for findings in a memory-off session.

**Search results include weird text that doesn't look like a document.** It's probably an agent-authored summary or finding being surfaced because the agent passed `--summaries` or `--findings` to `search`. Decide whether that's what you want; the default excludes both.

---

## License

MIT.