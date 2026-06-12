# Record the pinned embedding model in `meta` — backfill without a schema bump (issue #517)

> Source: [#517](https://github.com/jswest/bartleby/issues/517) — part of omnibus
> [#494](https://github.com/jswest/bartleby/issues/494) (v0.9.8).

A corpus's chunk embeddings are produced by the model pinned in
`bartleby/lib/consts.py::EMBEDDING_MODEL` (`BAAI/bge-base-en-v1.5`). Until now the
DB carried `meta.embedding_dim` but not *which* model produced those vectors, so a
query embedded with a different model than the corpus could silently retrieve
garbage. Recording the model name is the foundation; the import-side
*verification* that compares a corpus's recorded model against the running pin is
[#520](https://github.com/jswest/bartleby/issues/520), not this issue.

Two populations to cover:

- **New corpora.** `init_db` already writes `('embedding_model', EMBEDDING_MODEL)`
  into `meta` alongside `embedding_dim` and the other init rows — this predates
  #517 (it landed with the v1 migration). Nothing to add; a regression test now
  pins it to `EMBEDDING_MODEL` verbatim rather than the looser truthiness check.
- **Existing corpora.** A corpus created before that init row existed has no key.
  It gets the *current pinned* value backfilled when the user runs
  `bartleby project upgrade <name>`.

## The stays-at-9 mechanism (and why no version step)

This omnibus is explicitly additive with `SCHEMA_VERSION` pinned at **9** — the
release is `v0.<SCHEMA_VERSION>.<patch>` = v0.9.8. A literal `_upgrade_v9_to_v10`
chain entry would only ever fire if `SCHEMA_VERSION` became 10, which the omnibus
forbids; adding one would be dead code at best and a forced re-ingest signal at
worst. So the backfill must run *without* a version bump.

`upgrades.py::upgrade()` already has the right seam: a tail block that runs on
**every** `project upgrade`, even when zero version steps fire (it stamps
`upgraded_at`). The backfill is a single idempotent statement added there:

```sql
INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_model', <pinned>)
```

`INSERT OR IGNORE` is what makes this safe to attach to an unconditional,
already-at-v9 path:

- A corpus **missing** the key gets the current pinned value.
- A corpus that **already carries** a value keeps it — the backfill never
  overwrites, so a corpus built on a different model is not silently relabelled
  (that mismatch is #520's job to *detect*, not this code's to paper over).
- Re-running `project upgrade` is a no-op — no churn, no version movement.

No `schema_version` is touched by this statement; the column-walk loop above is
untouched. The DB-newer-than-code guard still raises *before* the tail block, so
the backfill never runs on a DB this code can't reason about
(`test_upgrade_refuses_db_newer_than_code` still asserts the tail never fired).

## Rejected alternative

Adding `_upgrade_v9_to_v10` + bumping `SCHEMA_VERSION` to 10. This is the
mechanically "canonical" upgrade-chain shape, but it turns v0.9.8 into v0.10.0 — a
minor/schema bump that is the maintainer's release call, not an in-omnibus
detail — and it would mark every existing v9 corpus as out-of-date for a purely
additive metadata write. Rejected: the idempotent tail backfill achieves the same
result with zero version movement.

## Tests (in `tests/test_db.py`)

- `test_init_records_pinned_embedding_model` — fresh DB's `meta.embedding_model`
  equals `EMBEDDING_MODEL`.
- `test_upgrade_backfills_embedding_model_idempotently` — delete the key to
  simulate a pre-#517 corpus, run `upgrade(conn, SCHEMA_VERSION)` (no version step
  fires; only the tail runs), assert the pinned value lands, and assert a second
  pass leaves exactly one unchanged row.
- `test_upgrade_does_not_overwrite_existing_embedding_model` — a pre-existing
  non-default value survives an upgrade (INSERT OR IGNORE semantics).
- `test_schema_version_pinned_at_nine` — guards the omnibus's additive contract.
