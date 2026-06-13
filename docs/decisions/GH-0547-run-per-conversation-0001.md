# One run per conversation, via an agent-minted run_key

**Supersedes [GH-XXXX — Sessions auto-create](GH-XXXX-sessions-auto-create-0001.md).**

## Problem

A research "run" (a `sessions` row) was sticky: tracked by a single
`.active_session` pointer per corpus that persisted across everything — new
agent conversations, `/clear`, days. Nothing a person naturally does started a
fresh run, so unrelated efforts piled into one session, and the recorded model
could be stale. That blocks the findings view's reason to exist: grouping and
comparing runs (e.g. Opus vs. a local model over the same corpus).

## The call

A run is bound to one conversation through a UUID the agent mints itself, with
**zero human setup**:

- The agent's first action each conversation is `bartleby skill session new`,
  which mints a `run_key` (uuid4), creates the run, and returns the id. The
  agent carries it as `--run <uuid>` on every later call.
- The runner resolves a session in order: `BARTLEBY_SESSION_NAME` (the web's own
  pinned session) → `--run` (get-or-create the run bound to that key) → the
  `.active_session` marker (the fallback when a call forgot `--run`).
- Per-run records (one row per `run_key`, a UNIQUE index) — not one shared
  marker — are the concurrency fix: two conversations on one corpus get distinct
  runs that never overwrite each other. The marker survives only as the
  forgot-my-id safety net, and every result echoes the current run back so the
  agent can recover its `run_key` and notice a wrong-run fallback.
- The model is **self-reported** by the agent (`session new --model`),
  best-effort, surfaced everywhere as "Set by LLM" — a claim, not a verified
  fact — and left blank when the agent doesn't know its own name (common for
  local models). No env vars, no flags the human must set.

## Why this shape

The agent's context window is the one thing that is reliably per-conversation,
needs no human setup, and the agent always has — so the run id lives there and
everything keys off it. New conversation → new `session new` → new run; `/clear`
→ fresh context → new run; concurrent conversations → distinct keys → no
collision. Continuity across runs is already carried by the project DB + memory
(a memory-on run sees prior findings), so ending a run loses nothing — the
session never needed to be the long-lived layer.

Prompt-adherence is accepted as flaky: a model that skips `session new` or drops
`--run` falls back to the marker, which is usually the right run. The user chose
this over any scheme that adds human friction (env vars, a launcher, manual
`session start`) — "the human should not have to do anything," flakiness
accepted.

## Schema

Additive: `sessions` gains a nullable `run_key TEXT` plus a UNIQUE index (the
index, not a column constraint, so the many NULL keys on CLI/web/legacy sessions
coexist — and `ALTER TABLE` can't add a column-level UNIQUE anyway). Bumps
`SCHEMA_VERSION` 9 → 10 with a `_upgrade_v9_to_v10` chain entry, so existing
corpora `bartleby project upgrade` rather than re-ingest. This is the literal
`_upgrade_v9_to_v10` that [GH-0508](GH-0508-record-embedding-model-0001.md)
deliberately *declined* to add for the embedding-model backfill (that one had to
stay at v9); run_key is a genuine schema addition, so the bump is earned here.

## Rejected

- **A `model_source` column** to mark the model as LLM-set — unnecessary; an
  agent-minted run (run_key present) is the only way a model gets self-reported,
  so "Set by LLM" is derived from `run_key IS NOT NULL AND model IS NOT NULL`. One
  new column, not two.
- **Env-var / launcher / SessionStart-hook propagation** of a run id — all add
  human or harness setup; non-CC backends are first-class, and the agent-minted
  UUID needs none of it.
- **A single shared "current run" marker** as the primary mechanism — loses to
  concurrency (two conversations clobber each other). Kept only as the fallback.

## Known interaction (not resolved here)

A human who pre-starts a memory-off session (`bartleby session start
--no-memory`) for eval isolation no longer governs a SKILL.md-following agent,
which mints its own memory-on run via `session new`. `session new --no-memory`
keeps memory-off *expressible*, but how a human reliably gets a memory-off agent
run in the new flow is left open for a follow-up.
