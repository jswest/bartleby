# GH-0357: Fix schema-mismatch test direction

## Decision

`test_schema_mismatch_points_at_upgrade_first` now stamps `SCHEMA_VERSION - 1`
(an **older** DB) before reopening, instead of `SCHEMA_VERSION + 1` (a newer DB).

## Why

`open_db` raises on any `db_version != SCHEMA_VERSION` and always emits the same
remedy: "run `bartleby project upgrade`, and if that reports a non-additive bump,
re-ingest." That advice only makes sense for an **older** on-disk DB — the
in-place upgrade chain (`bartleby/db/upgrades.py`) walks `vN → vN+1` forward, so
it can only bring an older corpus up to the code's expectation.

The old test stamped `SCHEMA_VERSION + 1`, simulating a DB *newer* than the code.
That direction is semantically the opposite: you cannot "upgrade" a database that
is already ahead of the code — the correct remedy there is "update the code, not
the DB." Asserting the upgrade-first message against a newer DB tested the message
in a direction where it does not actually apply. Stamping `SCHEMA_VERSION - 1`
exercises the older-DB direction the upgrade-first remedy is written for.

The newer-DB "update the code, not the DB" path (CLI `upgrade()` guard) is covered
by a separate issue and intentionally not duplicated here.

## Scope

Test-only change in `tests/test_db.py`. No production code, no schema change,
`SCHEMA_VERSION` untouched.
