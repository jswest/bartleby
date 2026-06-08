# Complete the v7→v8 additive upgrade rather than force a re-ingest (issue #212)

> Source: [#212](https://github.com/jswest/bartleby/issues/212)

supersedes the upgrade-handling plan in
[GH-0171](GH-0171-unit-ingest-provenance-config-drift-warnings-0001.md).

#171 deferred the upgrade-chain entry for the `ingests` table + `ingest_run_id`
columns to "one consolidated `_upgrade_v8_to_v9` at the #169 release". That entry
never materialized the way it was planned: the concurrent-ingestion omnibus
shipped *staying at* schema v8 (no v8→v9 bump), and `_upgrade_v7_to_v8` — added by
#164 when v8 meant only `failed_ingests` — was never extended to cover the
provenance DDL #171 later folded into v8. The result was a half-additive upgrade:
`project upgrade` on a v7 corpus stamped the DB v8 while omitting `ingests` and
the three `ingest_run_id` columns. Because *no read path* touches provenance, the
DB read green until the first write — and every chunk insert unconditionally names
`ingest_run_id`, so `save_finding` / `save_summary` / re-running `scribe` all
raised `no such column: ingest_run_id`. An upgrade entry *existing* meant
`project upgrade` didn't refuse; it ran the partial migration and reported
success — exactly the silently-half-migrated-reads-green trap the resumable-ingest
work set out to kill.

Two directions were weighed:

- **A — complete the additive upgrade (chosen).** The missing DDL is genuinely
  additive (a new table + nullable FK columns), so finish `_upgrade_v7_to_v8`:
  `CREATE TABLE ingests` + `ALTER TABLE {documents,summaries,chunks} ADD COLUMN
  ingest_run_id INTEGER REFERENCES ingests(run_id)`. SQLite permits the inline FK
  on `ADD COLUMN` because the column is nullable with no non-NULL default; columns
  append at end-of-table so an upgraded DB is schema-equivalent to a fresh v8.
  Old rows carry `ingest_run_id = NULL`, which the chunk helpers already expect.
  No `SCHEMA_VERSION` change — stays v8, patch-level (fits v0.8.1).
- **B — delete the v7→v8 entry so `project upgrade` refuses → re-ingest.** Cleaner
  if provenance is held to be strictly re-ingest-only, but it strands v7 corpora:
  re-ingesting every file (multi-hour on a real corpus) only to reach a working v8
  whose sole gap is retroactively-unknowable provenance. Rejected — the DDL is
  additive and the only real harm is the missing-column crash, which A fixes
  directly.

The pre-upgrade "loss" under A is that rows ingested before the upgrade carry no
provenance — inherently unknowable retroactively, the same NULL state #171 already
defined for pre-feature units. The lockstep between this chain and `db/schema.py`
is guarded by `tests/test_project.py::test_upgrade_chain_walks_from_v4_through_current`,
extended here to drop the v8 provenance when simulating v4, assert it returns,
check the upgraded DB is schema-equivalent to a fresh v8, and regression-test
saving a finding on an upgraded DB (the exact write that used to crash).
