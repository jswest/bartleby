# Guard `upgrade()` against a DB newer than the code (issue #354)

> Source: [#354](https://github.com/jswest/bartleby/issues/354)

`upgrades.upgrade()` walked the chain from `current_version` up to
`SCHEMA_VERSION` but never guarded the case where `current_version >
SCHEMA_VERSION`. The `while v < SCHEMA_VERSION` loop simply never ran, fell
through to the unconditional trailing `with conn:` block, and stamped
`upgraded_at` on a DB this code can't reason about. Worse, the contract was that
the library was the only place callers reached — yet only the CLI wrapper
(`commands/project.py`) caught a newer DB (and refused with exit 1). The
library itself had the hole, and that CLI guard was untested.

**Fix: refuse a newer DB at the top of `upgrade()`, before any mutation.** A
single early guard raises a `RuntimeError` when `current_version >
SCHEMA_VERSION`:

```
DB schema vN is newer than this code (vM). Update the code, not the DB.
```

The wording matches the existing CLI guard in `commands/project.py`
("Update the code, not the DB.") so the two surfaces speak the same language.
The guard runs before the chain walk and before the trailing `upgraded_at`
write, so nothing is stamped or mutated. The CLI catch (#353 widened it to
`(RuntimeError, apsw.Error)`) surfaces it as a readable `Upgrade failed: <e>`
exit-1 line for any caller that reaches the library directly; the CLI's own
pre-check still short-circuits the common path with its dedicated message.

**Test: `test_upgrade_refuses_db_newer_than_code` (`tests/test_project.py`).**
It creates a fresh project, stamps `meta.schema_version` to `SCHEMA_VERSION + 1`,
then asserts (a) the library `upgrade()` raises `RuntimeError` matching
"newer than this code", (b) `meta` is byte-for-byte unchanged afterward —
`schema_version` is *not* rewritten down and no `upgraded_at` key appears — and
(c) the CLI wrapper `project_cmd.upgrade` refuses with `SystemExit(1)`. The
no-mutation assertion is the load-bearing one: it pins the exact corruption the
issue describes (a newer DB silently downgraded).

Not done, deliberately: `SCHEMA_VERSION` is untouched (held at 8), and neither
the dormant `_upgrade_v8_to_v9` body nor the held-at-8 xfail on the chain-walk
gate was altered — those belong to the v0.9.0 assembly commit.

touches: `bartleby/db/upgrades.py`, `tests/test_project.py` (one added test).
No schema change.
