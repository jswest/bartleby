# Pin the upgrade-chain refusal path: a DB below the first step is refused, DB unchanged (issue #356)

> Source: [#356](https://github.com/jswest/bartleby/issues/356)

Part of the v0.9.0 additive schema-bump omnibus ([#363](https://github.com/jswest/bartleby/issues/363)).

## Problem

`upgrade()` walks `_UPGRADES` from the DB's stamped version up to `SCHEMA_VERSION`,
raising a `RuntimeError` ("non-additive bump; re-ingest is required") the moment it
hits a version with no chain step (`upgrades.py` `_UPGRADES.get(v) is None`). The CLI
wrapper (`commands/project.py`) catches that and `sys.exit(1)`s. Neither path had a
test: the refusal `RuntimeError` and the CLI `exit(1)` were both uncovered, and the
load-bearing **"a refused upgrade leaves the DB unchanged"** property was unpinned.

That property matters specifically because the chain now stamps `meta.schema_version`
*inside each step's own transaction* (the per-step-commit structure from #353/#354's
mid-chain-crash work). A future mid-chain hole would persist the earlier steps' DDL
while leaving the stamp old — so we want an explicit guarantee that a *refused* walk
(one with no step to begin from) writes nothing at all.

## What the test does

`test_upgrade_refuses_below_chain_first_step_without_mutation` (in
`tests/test_project.py`):

- Creates a fresh project, then stamps `meta.schema_version` to
  `min(_UPGRADES) - 1` (currently v3 — one below the chain's first entry, v4). No
  production code branches on schema version, so the stale stamp alone reproduces
  the "below the chain, nothing to walk from" condition without hand-mutating DDL.
- Snapshots the full `sqlite_master` (type, name, tbl_name, sql) **and** the
  `schema_version` stamp before the call.
- Invokes the CLI `project_cmd.upgrade(name=...)`, asserts it raises `SystemExit`
  with code 1 and that the rendered output contains the re-ingest guidance.
- Re-snapshots and asserts the snapshot is **byte-identical** before/after — both
  the full DDL and the version stamp — so the refusal provably mutated nothing.

## Why v3, not "newer than code"

`test_upgrade_refuses_db_newer_than_code` already covers `current_version >
SCHEMA_VERSION` (the tail `upgraded_at` write that would rewrite a newer stamp down).
This test covers the *other* refusal branch — `current_version < SCHEMA_VERSION` with
a hole at the bottom of the chain — which is the path the issue calls out as
uncovered. Stamping below `min(_UPGRADES)` (computed, not hard-coded to 3) keeps the
test honest if the chain's floor ever moves.

## Scope

Test-only. No production change, no `SCHEMA_VERSION` bump, and the held-at-8 xfail on
`test_upgrade_chain_walks_from_v4_through_current` is untouched. Full suite stays at
green + 1 xfailed. The refusal path was verified to genuinely leave the DB unchanged
(no real mutation bug found).
