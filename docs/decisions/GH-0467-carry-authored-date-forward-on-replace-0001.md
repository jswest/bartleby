# `save_summary` carries `authored_date` forward on replace (issue #467)

> Source: [#467](https://github.com/jswest/bartleby/issues/467)

`save_summary` does a full-row REPLACE: it deletes the prior summary (and its
chunks) and inserts a fresh row. Because `--authored-date` defaulted to `None`
and was always normalized straight into the INSERT, *re-saving* a dated summary
without re-passing the date silently wrote NULL — dropping the document out of
every authored-date scope (`--authored-after`/`--authored-before`, the
`authored_date` that `list_documents`/`scan`/`search` report). That is a
data-loss footgun for the common "fix the title/description, keep everything
else" re-save.

The original issue offered two options: (a) a doc-only line warning that a
re-save nulls the date, or (b) a behavior fix that carries the prior date
forward. The director took the behavior flip (2026-06-11): the silent drop is a
correctness bug, not a documentation gap, and the replace path already SELECTs
the prior row — carrying the date forward is the smallest fix that actually
removes the footgun.

The fix is at the one replace site: the prior-row SELECT now also fetches
`authored_date`, and when `--authored-date` is omitted **and** a prior row
exists, that prior date is bound into the INSERT instead of NULL. Three cases:

- **Replace, no `--authored-date`** → prior date carries forward.
- **Brand-new save, no `--authored-date`** (no prior row) → stays NULL, as
  before. Carry-forward never fabricates a date out of nothing.
- **Explicit `--authored-date`** → always overwrites (still normalized; a
  malformed value still stores NULL). To set or change *only* the date,
  `save_date` remains the dedicated path (it owns `--clear`); this fix
  deliberately does **not** add a new clear flag to `save_summary`.

Bounded to the write-path: no schema change (a behavior fix on an existing
column, no DDL), no redesign of date handling, no new abstraction. Writes still
go through the typed chunk helpers. Help text, the module docstring, and the
`save_summary` row in `SKILL.md` now state the carry-forward-on-replace
behavior; a test re-saves a dated summary with no `--authored-date` and asserts
the date survives, with companion tests pinning explicit-overwrite and
new-save-stays-NULL.
