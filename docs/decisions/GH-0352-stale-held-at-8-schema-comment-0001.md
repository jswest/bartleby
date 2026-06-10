# Resolve the stale "held at 8" schema comment + the v8→v9 authoring rule (issue #352)

> Source: [#352](https://github.com/jswest/bartleby/issues/352)

The header comment on `SCHEMA_VERSION` in `bartleby/db/schema.py` still described a
plan that never happened. It promised that "a consolidated `_upgrade_v8_to_v9`
ships once when #169 releases" and told existing v8 corpora to **re-ingest** to gain
the `ingests` table / `ingest_run_id` columns. Both halves were wrong by the time
v0.8.x shipped:

- #169 is **closed** (released as v0.8.x), and the omnibus shipped *staying at* v8 —
  no v8→v9 bump, no consolidated `_upgrade_v8_to_v9`.
- `_upgrade_v7_to_v8` was completed (issue
  [#212](https://github.com/jswest/bartleby/issues/212),
  [GH-0212](GH-0212-complete-v7-to-v8-additive-upgrade-0001.md)) to create
  `ingests`/`ingest_run_id` **additively**, so a v7 corpus reaches the full v8 shape
  via `bartleby project upgrade` — not re-ingest.

Two live hazards motivated writing this down rather than just deleting the stale
text:

1. **Forward trap.** A naive `_upgrade_v8_to_v9` that re-shipped the `ingests` DDL
   would crash every released v0.8.x DB with "table ingests already exists",
   because v7→v8 already creates it.
2. **Window cohort.** A DB created from `main` between #164 (stamped v8 with only
   `failed_ingests`) and #171 (folded `ingests`/`ingest_run_id` into v8, still v8)
   carries `v8` in `meta` without those structures. `open_db` accepts the version
   number, then the first write crashes raw on the missing column. There is no
   upgrade step that can repair it — it shares a version number with released v8 but
   not its DDL.

## Decision

Comment/doc only — no behavior, schema, or `SCHEMA_VERSION` change.

- **`schema.py` header** now declares: **v8 == the with-`ingests` shape** (the
  `ingests` table + `ingest_run_id` columns are *part of* v8, not deferred); the
  #169 omnibus shipped staying at v8 and the promised consolidation never
  materialized; and the **#164–#171 window cohort is re-ingest-only**. Released
  v0.8.x DBs already carry the full v8 shape and are unaffected.

- **`upgrades.py` module docstring** records the authoring rule for any
  `_upgrade_v8_to_v9`: it is written against **released v0.8.x DBs** (which already
  contain `ingests`/`ingest_run_id`), so it must NOT re-create those and must be
  **additive-only**. The window cohort is explicitly out of scope (re-ingest-only).

This entry is consistent with the now-written versioning policy
([GH-0362](GH-0362-schema-change-versioning-policy-0001.md)): additive-or-breaking
is binary, and an additive bump ships a chain entry. It documents the *durable*
versioning truth and the authoring rule for the eventual v8→v9, independent of the
transient "held at 8" state of the v0.9.0 omnibus (where the v9 columns are present
in `schema.py` and the `_upgrade_v8_to_v9` step is dormant until the assembly commit
bumps `SCHEMA_VERSION` to 9 and removes the held-at-8 xfail on the chain-walk test).
The dormant `_upgrade_v8_to_v9` function body and the v9 columns are left untouched
by this change.
