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

The skill ships inside the `bartleby` package, so the CLI installs it for you.

**Claude Code:**

```
bartleby ready
```

This stamps the skill into `~/.claude/skills/bartleby/`. Re-run it any time to refresh; `bartleby ready --check` reports whether your installed copy is current.

**Other harnesses:** point it elsewhere with `bartleby ready --dest <dir>`. The skill is a self-contained folder; it drops anywhere a compliant harness reads skills from.

The main [README](../../README.md#install-the-skill) covers refreshing after an update — worth reading if you're updating an existing install.

---

## What the skill exposes

The skill ships a `scripts/` directory containing small Python scripts that the agent calls. Each takes arguments, prints JSON to stdout, and exits non-zero on error.

| Script | Purpose |
| --- | --- |
| `describe_corpus` | One cheap pure-SQL aggregate overview — counts, `authored_date` range + undated count, per-year histogram, tag distribution, summary coverage, content-type mix, and top-N largest documents. The recommended first call on an unfamiliar corpus. |
| `list_documents` | Enumerate documents in the corpus (file names, IDs, page/token/chunk counts, summary status). |
| `search` | Unified search across documents, summaries, and findings. Supports keyword (FTS5), semantic (vector), and hybrid (RRF) modes. Hits return just the matched chunk by default; `--add-context N` (0..5) attaches N neighbor chunks on each side. |
| `scan` | Full-text-only *filter* (no ranking): returns every document chunk matching a literal phrase corpus-wide, in document + source order, paginated with a true `total`. For enumerating marker phrases on templated corpora; documents only, compact snippets by default. `--match-terms` switches phrase matching to a boolean AND of tokens. A quoted phrase matches a token *sequence* and FTS5 treats punctuation as a boundary, not a token, so the phrase reaches across intervening punctuation (`"foo bar"` matches `foo, bar` / `foo. Bar`) — handy for pinning a templated string through its brackets/commas; you cannot anchor on the punctuation itself. |
| `read_chunks` | Read a window of chunks from a document. Paginated via `--offset` and `--limit`. |
| `read_document` | Read a full document and/or its summary. Refuses oversized documents without `--force`. |
| `save_summary` | Save an agent-authored summary back into the database (chunked and embedded). |
| `save_finding` | Save a finding (markdown text + structural citations) into the database. |
| `merge_findings` | Collapse a cluster of duplicate findings into one. The `--into` target survives (keeps its id); you author the consolidated body via `--body-file`; the `--from` sources are deleted. The curation counterpart to `merge_tags`. |
| `delete_finding` | Retract a finding outright — its row, body chunks, and citations. Cited document chunks (evidence) are untouched. The curation counterpart to `delete_tag`. |
| `list_findings` | Browse prior findings (newest first): id, title, description, authoring session, created-at, citation count. Paginated. The enumeration counterpart to `search --findings`. |
| `read_finding` | Read one whole finding by id — full body, the finding's chunks, and resolved citations. Same shape as `save_finding`. |

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

Beyond relevance search, findings have two direct read paths: `list_findings` enumerates them (newest first, for browsing), and `read_finding --finding <id>` returns one whole finding. In a memory-off session these don't fail outright — they scope to the session's **own** findings (so a run can read back what it just wrote): `list_findings` lists only this session's findings, and `read_finding` returns this session's findings but raises `{"code": "MEMORY_OFF"}` for one authored by another session. `search` is stricter — it silently drops **all** findings regardless, since ranked retrieval surfacing a finding as if it were evidence is the contamination path memory-off exists to prevent.

Findings also have a curation path so memory can be tended rather than only grown: `delete_finding --finding <id>` retracts one (its row, body chunks, and citations), and `merge_findings --from <ids> --into <id> --body-file <path>` folds a cluster of duplicate iterations into a single consolidated finding (the target survives, the agent authors the merged body, the sources are deleted). Both touch only finding rows — the cited document chunks are never affected, because findings are derivative hints, not evidence.

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

If you want different defaults, edit `SKILL.md`.

---

## Troubleshooting

**"Schema version mismatch."** The database was created by a different version of the `bartleby` CLI. Either upgrade the CLI or re-ingest the corpus.

**"No active project."** Run `bartleby project use <name>` or pass `--project` via your harness's environment.

**Agent is calling tools but getting empty results.** Check `bartleby logs` for what it actually queried. The most common cause is searching for findings in a memory-off session.

**Search results include weird text that doesn't look like a document.** It's probably an agent-authored summary or finding being surfaced because the agent passed `--summaries` or `--findings` to `search`. Decide whether that's what you want; the default excludes both.

---

## License

MIT.