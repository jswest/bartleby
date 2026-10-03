# The Bartleby Skill

A skill that lets an agent romp through a Bartleby corpus--searching, reading, synthesizing, citing, and saving findings--inside any compliant harness (Claude Code, Cowork, or similar).

The skill is BYO-model. It works against whatever model your harness runs.

---

## What this is, and what it isn't

**It is:** a set of small Python scripts that talk to a Bartleby SQLite database, plus a `SKILL.md` that tells the agent how to use them well. The skill is opinionated — it has views about what counts as evidence, when to search vs. read, and how to behave when memory is on or off.

**It isn't:** a way to ingest documents. That's the [`bartleby` CLI](../../README.md). The skill assumes the database already exists and the corpus is already chunked, embedded, and indexed.

---

## Prerequisites

1. The `bartleby` CLI is installed and on your `PATH`. The skill shells out to it for embedding queries (semantic search). Everything else (config, schema, project resolution) is imported from the installed `bartleby` package directly. (See the [main README](../../README.md) for install instructions.)
2. A Bartleby project exists, with documents already ingested (`bartleby scribe`).
3. The project is the active project, *or* each script is invoked with `--project <name>`.

Every script opens the project DB via the shared runner, which validates the schema version on the spot and refuses to run against an incompatible database. The agent opens its own run on its first call — no session setup on your part (see "How runs work" below).

---

## Installation

Run [`bartleby ready`](../../README.md#bartleby-ready): it installs to `~/.claude/skills/bartleby/` (`--dest <dir>` for other harnesses, `--check` to see if it's current). The skill is a self-contained folder; it drops anywhere a compliant harness reads skills from.

---

## What the skill exposes

The agent calls these scripts via `bartleby skill <name>`. They're package-internal (under `bartleby/skill_scripts/`), dispatched by the CLI — the installed bundle itself ships only `SKILL.md` + `README.md`, not a `scripts/` directory. Each script takes arguments, prints JSON to stdout, and exits non-zero on error.

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
| `annotate_finding` | Attach a note to a finding without rewriting it — whole-finding or anchored to a verbatim quoted span, optionally pointing at a chunk. Memory-gated on the parent finding. |
| `delete_annotation` | Remove one annotation by id (no in-place edit — delete and re-annotate). Memory-gated on the parent finding. |
| `list_findings` | Browse prior findings (newest first): id, title, description, authoring session, created-at, citation count, annotation count. Paginated. The enumeration counterpart to `search --findings`. |
| `read_finding` | Read one whole finding by id — full body, the finding's chunks, resolved citations, and annotations (oldest first). Same shape as `save_finding`. |
| `edit_finding` | Update an existing finding's title, description, and/or body. Memory-gated like the other finding reads/writes. |
| `save_date` | Backfill or correct a document's `authored_date` (the date-only counterpart to re-saving a summary; supports `--clear`). |
| `extract` | Run a value-tag's stored regex over a set of chunks, storing the per-document values it captures. |
| `read_tags` | List the controlled tag vocabulary — names, descriptions, and document counts. |
| `add_tag` | Create a new tag in the controlled vocabulary (optionally a value-tag with a `value_type` + regex `pattern`). |
| `rename_tag` | Rename a tag; assignments preserved. |
| `delete_tag` | Remove a tag from the vocabulary. The curation counterpart to `delete_finding`. |
| `merge_tags` | Move all assignments from one tag (`--from`) onto another (`--into`), then delete the source. The tag counterpart to `merge_findings`. |
| `tag` | Classify documents against the controlled vocabulary (LLM-assisted) — runs the configured summarizer model once per document; `tag --all` requires explicit human confirmation. |
| `assign_tag` | Attach one tag to one or more documents directly, bypassing the classifier. |
| `unassign_tag` | Detach one tag from one or more documents. |

The scripts wrap a shared Python library that owns all writes to the chunks table. Source-kind discipline (`document` vs. `summary` vs. `finding` vs. `image`) is enforced both by a `CHECK` constraint at the SQL layer and by typed insert helpers in the library, so agents can't accidentally mislabel chunks.

For full argument-level contracts, see [`SKILL.md`](./SKILL.md) and the script docstrings.

---

## How runs work

Every agent conversation is one *run*. Runs are rows in the `sessions` table with an ID and a memorable name (e.g., `mighty-grove`); findings and audit log entries are tagged with the run's `session_id`.

**The agent opens its own.** Its first call is:

```
bartleby skill session new [--model <id>] [--no-memory]
```

This mints a `run_key` (a UUID) and starts a fresh run. The agent passes `--run <run_key>` on every later call so its work attaches to that run, and every successful result echoes the run back under `"run"`. A new conversation is a new run; two conversations on one corpus never share one. A call that forgets `--run` falls back to the most recently used run.

Runs don't really "end" — there's no end-state to enforce. They group related work and thread provenance through the database.

**Memory:**

By default a run is memory-on: `search --findings`, `list_findings`, and `read_finding` reach findings from any prior run, so the agent can build on past research.

To have the agent ignore prior findings, ask for it in the **first message** of a new conversation ("ignore previous memory"). The agent opens its run with `bartleby skill session new --no-memory`. In a memory-off run, `search` silently drops all findings from results, *regardless of what flags the agent passes*, and the direct finding reads and curation commands reach only the run's own findings. This is enforced at the script level, not via prompt. The agent literally cannot reach prior findings.

Memory-off is fixed when the run opens. If you ask mid-conversation, the skill instructs the agent to stop and tell you to start a new conversation and ask up front — it won't switch runs mid-stream, since what it has already read stays in its context.

The human `bartleby session` CLI inspects and labels runs but doesn't drive them — see the [main README](../../README.md#bartleby-session).

---

## How findings work

Findings are the durable output of a research session. They live in the `findings` table as markdown text, *plus* a `finding_citations` join table linking each finding to the source chunks it rests on.

Findings are chunked and embedded into the same vector space as documents and agent-generated summaries. This means cross-session memory is just semantic search — the agent searches its own past findings the same way it searches the corpus.

Findings are tagged with `source_kind = 'finding'` and excluded from search by default. The agent must opt in via `--findings` to include them. The skill's prompt guides the agent on when to do this (typically: at the start of a new topic, to check for prior relevant work; never as primary evidence in a citation).

Beyond relevance search, findings have two direct read paths: `list_findings` enumerates them (newest first, for browsing), and `read_finding --finding-id <id>` returns one whole finding. How memory-off narrows these is under "How runs work" above.

Findings also have a curation path so memory can be tended rather than only grown: `delete_finding --finding-id <id>` retracts one (its row, body chunks, and citations), and `merge_findings --from <ids> --into <id> --body-file <path>` folds a cluster of duplicate iterations into a single consolidated finding (the target survives, the agent authors the merged body, the sources are deleted). Both touch only finding rows — the cited document chunks are never affected, because findings are derivative hints, not evidence.

---

## What's stored, and where

Everything queryable lives in the project's `bartleby.db`:

- `documents` — one row per ingested file
- `chunks` — polymorphic: documents, summaries, findings, and images all chunk into here, tagged with `source_kind` (`document` / `summary` / `finding` / `image`) and `source_id`
- `summaries` — one row per document (single-shot, whole-document)
- `sessions` — one row per agent session
- `findings` — one row per saved finding, with markdown text
- `finding_citations` — join table from findings to the chunks they cite
- `tags` — the controlled tag vocabulary: name, description, and (for value-tags) a `value_type` + capture `pattern`
- `document_tags` — join table assigning tags to documents (with the captured `value` and source `chunk_id` for value-tags)
- `audit_logs` — every tool call the agent made, append-only, never read by the agent

No sidecar files. To share a finding as a file, use `bartleby finding export` (see the [main README](../../README.md#bartleby-finding)).

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

**"Schema version mismatch."** The database was created by a different version of the `bartleby` CLI. Run `bartleby project upgrade <name>`. If it says the DB is newer than your code, update the CLI instead; if it otherwise refuses, re-ingest — see the [main README](../../README.md#after-a-schema-change).

**"No active project."** Run `bartleby project use <name>` or pass `--project` via your harness's environment.

**Agent is calling tools but getting empty results.** Check `bartleby logs` for what it actually queried. The most common cause is searching for findings in a memory-off session.

**Search results include weird text that doesn't look like a document.** It's probably an agent-authored summary or finding being surfaced because the agent passed `--summaries` or `--findings` to `search`. Decide whether that's what you want; the default excludes both.

---

## License

MIT.