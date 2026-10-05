# The test suite is a bug-regression guard suite; the `simplify-refactor` gate is dropped (issue #745)

> Source: [#745](https://github.com/jswest/bartleby/issues/745)

Reverses two habits. It supersedes the gate half of
[GH-0335](GH-0335-ultraship-subagent-players-gate-at-stage-manager-0001.md), which
moved the `simplify-refactor` gate from the player up to the stage-manager. That
gate no longer exists anywhere. GH-0335's player mechanism (players are isolated
subagents) stands.

**Why.** The suite had grown to about 25k lines and 1,334 tests, more than the
21.6k-line source. It ran in about 27s, so speed was never the problem. The cost
was upkeep. Agents wrote a test for every flag, error code and branch, so every
rename or refactor had to update them. An audit of all 80 test files found roughly
25–30% outright padding. Most of the rest were end-to-end checks whose value had
never been shown. Only about 50 tests guarded something we know matters. We don't
do TDD, and policing agents to keep tests short isn't worth the effort.

**1. The per-commit gate is `uv run pytest` → commit.** The
`.claude/agents/simplify-refactor.md` agent is deleted, and `gate_agent` is gone
from `.claude/ship.toml`. The pressed `ship` skill never read `gate_agent`, so no
drawer change or re-press was needed. `/ship`'s correctness and simplicity critics
are unchanged; they are advisory review, not a gate.

**2. Tests exist only to stop real bugs coming back** (see `AGENTS.md` for the
rule's wording). It lives in `AGENTS.md`, `CONTRIBUTING.md`, and the
`guardrails` key of `.claude/ship.toml`, which the
pressed skill injects verbatim into every player prompt. That key was renamed from
`live_data_note`, which the skill never read; the live-data redline text is
unchanged.

**3. The suite shrinks to the guard suite.** What stays:

- tests that enforce the `ARCHITECTURE.md` load-bearing invariants. These are
  properties an agent can break silently while the output still looks fine: the
  memory wall, partial writes leaving no trace, the chunks chokepoint, the schema
  upgrade chain and drift check, and publish being findings-free and leaving the
  source DB byte-identical;
- regression tests for real past bugs, as listed in #745;
- repo tooling checks (`test_guard_main_write.py`, `test_skill_drift.py`).

Everything else is deleted, along with orphaned fixtures and helpers and the tests
for one-off `scripts/` backfills. The suite went from 1,334 collected tests to 106
(74 test functions), grouped by what they guard: `test_memory_wall.py`,
`test_partial_writes.py`, `test_schema_upgrade.py`, `test_share.py`,
`test_chunks_chokepoint.py`, `test_skill_regressions.py`,
`test_ingest_regressions.py`, `test_runtime_regressions.py`, and the tooling
checks.

Process and tests only. No product code path changes and no `SCHEMA_VERSION` bump.
