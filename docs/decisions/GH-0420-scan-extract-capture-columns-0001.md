# `scan --extract '/regex/'` — regex captures into a tidy column table (issue #420)

> Source: [#420](https://github.com/jswest/bartleby/issues/420)

`scan` gained `--extract '/regex/'` (repeatable): per matching document chunk it
emits one row carrying `chunk_id` / `document_id` / `file_name` plus every
pattern's capture group(s) as columns. The point is a **tidy table for the agent
to post-process** — explicitly *not* a query engine (the #48/#10 boundary): no
predicates, no joins, no aggregation. It shares one capture primitive with
`--count-by '/regex/'`, which is now extract-then-group-and-count over that same
primitive.

**Column naming is fixed at parse time, named-vs-positional.** A **named** group
`(?P<amount>...)` projects to a column `amount`; **bare** groups project to
positional `g1`, `g2`, ... numbered over *all* groups in pattern order (so
`/(?P<bill>\d+)-(\w+)/` → `["bill", "g2"]`, following `re`'s 1-based group
indices so a positional name never lands in a different group's slot). This lives
on `CaptureSpec` in `_common.py` — the single place the column contract is
expressed — and is the same naming both modes inherit.

**A non-matching pattern yields a null cell, never a dropped row.** A row is a
*chunk*, not a *match*: if a chunk matches the FTS query but a given `--extract`
pattern finds nothing in it, that pattern's column(s) are `null` and the row
survives. That's what makes several `--extract` patterns composable onto one
row (`CaptureSpec.extract_first` returns an all-`None` dict on a miss) and what
keeps `--extract` a projection rather than a filter — it can never *remove* a row
the scopes admitted.

**Only the first match per chunk is captured.** `--extract` is a projection (one
row per chunk); `--count-by` is the per-match fold. Capturing every match per
chunk would turn one chunk into many rows and quietly re-introduce the
match-enumeration that `--count-by` already owns — so the two modes split cleanly
on first-match-projection vs. all-matches-fold, sharing the compile/validate/
column machinery but not the row cardinality.

**Multiple `--extract` patterns must produce distinct column names**
(`EXTRACT_COLUMN_COLLISION`, naming the offending `column`). Two bare groups both
wanting `g1`, or two named groups sharing a name, would silently clobber a cell;
rather than guess which capture wins we reject and point at `(?P<name>...)` as
the disambiguator. Honest-null is fine (a real captured absence); a silent
overwrite is not.

**The shared machinery, and what stays per-mode.** `parse_capture_regex` (compile
the `/.../`, require ≥1 group, raise `INVALID_CAPTURE_REGEX` / `CAPTURE_NO_GROUP`
as JSON envelopes — never a traceback) and `_scan_chunk_texts` (the deadline-
guarded chunk-text walk, with the runaway rails renamed `COUNT_BY_*` →
`CAPTURE_*` since both modes share them) are common. `--count-by '/regex/'` buckets
on the spec's *first* column; `--extract` projects *all* columns. `--extract`
cannot combine with `--count-by` / `--preview` / `--brief` / `--returning`: the
capture columns *are* its projection, so `--returning` (which selects from a
fixed whitelist) has nothing to act on — its dynamic column set is the output
contract instead. Additive only: no schema change, no re-index.
