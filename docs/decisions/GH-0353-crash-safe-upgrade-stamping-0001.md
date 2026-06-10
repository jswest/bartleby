# Crash-safe per-step upgrade stamping + widen the CLI catch (issue #353)

> Source: [#353](https://github.com/jswest/bartleby/issues/353)

`upgrades.upgrade()` committed each `_upgrade_vN_to_vN+1` step in its own
`with conn:` but stamped `meta.schema_version` only *after* the whole chain. A
crash after an earlier step committed left the DB structurally at vN+1 while
`meta` still said vN. A re-run then replayed that step and died on
"table already exists" — an `apsw.SQLError`, not a `RuntimeError`. The CLI
(`commands/project.py`) caught only `RuntimeError`, so the replay surfaced as a
raw traceback; meanwhile `open_db`'s strict version check rejected the DB, so
the corpus was permanently wedged with no path back.

**Fix: stamp `schema_version` inside each per-step transaction.** The
`UPDATE meta SET value = ?` for `schema_version` now lives *inside* each step's
`with conn:`, set to `v + 1`, so the DDL and the version bump commit atomically
together. A crash between two steps therefore leaves the DB at the *last
completed step's* version (never a structural-vs-meta mismatch), and a re-run
of `bartleby project upgrade` resumes from there — the loop reads the current
version and walks only the remaining steps, so no committed step is ever
re-applied. `upgraded_at` stays a single stamp at the end of the chain (its only
job is recording when the upgrade finished; it carries no resumption semantics).

**CLI catch widened to `(RuntimeError, apsw.Error)`.** A step failure (missing
step → `RuntimeError`; DDL error → `apsw.Error`) now exits 1 with a readable
`Upgrade failed: <e>` line in the repo's existing `[red]...[/red]` style, never a
raw apsw traceback. `apsw.Error` is the base of `apsw.SQLError` et al., so the
catch covers the whole family.

**Test: `test_upgrade_resumes_after_mid_chain_crash` (`tests/test_project.py`).**
It strips a fresh DB back to v4 (mirroring the chain-walk test's teardown),
monkeypatches the v6→v7 step (`_UPGRADES[6]`) to raise `apsw.SQLError` *after*
v4→v5 and v5→v6 have committed, and asserts the DB comes to rest at
`schema_version == 6` with the v6 tables present and the v7 columns absent. It
then restores the real step, re-runs via `project_cmd.upgrade`, and asserts the
chain resumes from v6 and completes to `SCHEMA_VERSION` — proving both
resumption and no double-application (re-walking a committed step would raise
"table already exists").

Not done, deliberately: `SCHEMA_VERSION` is untouched (held at 8), and neither
the dormant `_upgrade_v8_to_v9` body nor the held-at-8 xfail on the chain-walk
gate was altered — those belong to the v0.9.0 assembly commit.

touches: `bartleby/db/upgrades.py`, `bartleby/commands/project.py`,
`tests/test_project.py` (one added test). No schema change.
