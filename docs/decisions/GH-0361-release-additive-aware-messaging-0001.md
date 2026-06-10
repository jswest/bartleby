# Make `release.py` additive-aware in its messaging (issue #361)

> Source: [#361](https://github.com/jswest/bartleby/issues/361)

`release.py` hard-coded "Existing corpora must be re-ingested" on *any* minor
bump — in the GitHub Release notes (`build_release_notes`), in the dry-run
summary line (`Schema: N (bumped → re-ingest)`), and in the module docstring's
claim that the minor "never moves without a re-ingest". That was true while
every schema bump was breaking, but #212 established the sanctioned exception:
an *additive* bump moves `SCHEMA_VERSION` while a `_UPGRADES` chain entry covers
each crossed step, so existing corpora run `bartleby project upgrade <name>`
instead of re-ingesting. v0.9.0 is exactly such a bump, so the unconditional
banner would have mis-published a re-ingest order the release didn't need.

This change is **messaging only**. The DDL-drift refusal gate `check_drift`
(DDL changed but `SCHEMA_VERSION` static) is correct and untouched — it gates a
*different* failure (a forgotten bump) and never reads the upgrade chain.

The disposition is binary and is read from the same source of truth the runtime
upgrade path uses, so the two can't disagree: `_UPGRADES` in
`bartleby/db/upgrades.py`. A new pure helper `upgrade_covers(schema_from,
schema_to)` returns True only when **every** step `v -> v+1` from the old minor
up to the new schema has a chain entry (`all(v in _UPGRADES for v in range(from,
to))`). A single missing step makes the whole bump breaking — matching
`upgrade()`, which raises on the first gap. A non-forward move
(`schema_from >= schema_to`) is never additive.

`build_release_notes` and the dry-run summary now branch on `upgrade_covers`:

- **Additive** → "Existing corpora upgrade in place: run `bartleby project
  upgrade <name>`." / `(bumped → upgrade in place)`.
- **Breaking** → the original "must be re-ingested." banner / `(bumped →
  re-ingest)`.

`<name>` is left as a literal placeholder: release notes are corpus-agnostic, so
they name the command, not a specific project.

### Alternatives weighed

- **Re-derive coverage from the DDL diff** instead of consulting `_UPGRADES`.
  Rejected: it would re-litigate "is this bump additive?" in a second place and
  could drift from what `project upgrade` actually does. `_UPGRADES` *is* the
  registry of additive steps; reading it keeps notes and runtime in lockstep.
- **Touch `check_drift`.** Explicitly out of scope — it's a different,
  correct gate.

### Tests

`tests/test_release.py` covers both notes branches (additive → `project upgrade
<name>`, no "re-ingested"; breaking → "re-ingested", no "project upgrade") by
monkeypatching `release._UPGRADES`, plus `upgrade_covers` directly for the
single-step, multi-step-complete, missing-step, empty-chain, and non-forward
cases. The pre-existing `test_notes_banner_on_schema_bump` (which assumed every
bump re-ingests) was replaced by these two disposition-specific tests.
