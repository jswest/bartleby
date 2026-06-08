# CLAUDE.md

Instructions for Claude working in this repo. Terse on purpose. Two companions:
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — load-bearing invariants and current state. Read it before changing anything structural. The decision log (the *why* behind past calls) now lives one-per-file under [`docs/decisions/`](./docs/decisions/).
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — the human narrative of the workflow below.

Bartleby ingests documents into per-project SQLite (FTS5 + sqlite-vec), then a
skill scripts an agent over that DB. Two surfaces, one DB: the `bartleby` CLI
(ingestion + helpers) and the skill scripts in `bartleby/skill_scripts/`.

## Working agreements

- **No backwards compatibility, by default.** Delete old code — no compat shims, no
  feature-flagged old paths, nothing left dormant. The one sanctioned exception:
  *additive-only* schema upgrades (new tables, indexes, nullable columns) get a
  `_upgrade_vN_to_vN+1` entry in `bartleby/db/upgrades.py` so existing corpora run
  `bartleby project upgrade <name>` instead of re-ingesting.
- **Schema bumps mean re-ingest.** A non-additive schema change bumps `SCHEMA_VERSION`
  in `bartleby/db/schema.py` and tells users to re-ingest. There is no automatic
  migration for those; don't write one. A bump that's additive in DDL but needs data
  populated at ingest is effectively non-additive — re-ingest-only.
- **Polymorphic chunks discipline.** All writes to the `chunks` table go through the
  typed helpers in `bartleby.db.chunks`. Never `INSERT` into `chunks` directly — the
  `CHECK (source_kind IN ...)` constraint and the typed helpers are both load-bearing.
- **Memory enforcement is at script level, not prompt level.** If a session has
  `memory_enabled=0`, `search` excludes prior-session findings regardless of flags or
  prompting. Don't soften this.
- **Skill scripts speak JSON.** Print results as JSON to stdout; exit non-zero on
  error with `{"error": ..., "code": ...}`. Prose and progress go to stderr only.
- **When unsure, stop and ask.** Most choices here have a prior conversation behind
  them. If something is ambiguous or uncovered, ask rather than wing it.

## Workflow rails

- **Never commit or push on `main`.** The `guard-main-write.sh` hook enforces this; a
  block means you're on the wrong branch. Issue work happens in a **sibling worktree**
  (`../bartleby-issue-<N>-<slug>`) — never nested in the repo, never `git checkout -b`
  on `main`.
- **Pre-commit gates, every commit, in order:** `uv run pytest` (must pass) →
  `simplify-refactor` agent over the touched files → apply what's worth taking →
  re-run `uv run pytest` → commit.
- **Use `/ship #<N>`** for the full issue→PR loop. Claude opens the PR with a
  `Closes #<N>` line; **a human merges** — never merge yourself.
- **Use `/release`** to cut a release — a deliberate, post-merge act on `main`,
  never per-issue. It dry-runs `scripts/release.py`, pauses for your OK, then
  `--tag --push`. The script owns the version math; never invent a version or
  reach around the guard hook with a bare `git tag`/`git push`.

## Tooling

- **`uv`** for dependencies (not pip/venv); run code with `uv run python`.
- Tests: **`uv run pytest`** (the whole suite).
