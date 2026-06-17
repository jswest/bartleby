# Legacy bare `[^N]` chunk citations are reconciled by a one-shot data backfill, not reader leniency (issue #642)

> Source: [#642](https://github.com/jswest/bartleby/issues/642)

## The problem

GH-0624 was a hard cutover: the inline chunk-citation marker moved from bare
`[^N]` to typed `[^chunk:N]`, every reader (the web finding view
`findings/[id]/+page.svelte`, the `bartleby finding export/import` path) was
pointed at the prefixed grammar, and the save-time validators began *rejecting*
bare `[^N]`. But — per its own "interface-layer only, storage stays integer"
scope — it left existing `findings.body` text untouched and migrated nothing.

The result surfaced as #642: clicking an inline citation did nothing. A survey of
every finding across all project corpora found **2101 bare `[^N]` markers across
153 of 161 findings**; only the `test631` project (8 markers) used the new form.
A bare marker never matches the reader's `[^<scheme>:<ref>]` regex, so it renders
as inert prose with no dagger to click and no source loaded in the right pane.
The typed external `[^url:…]`/`[^doc:…]` markers (#563, predating #624) rendered
fine — the breakage was *only* the chunk citations. The click handler and the
`SourceViewer` right pane were never broken.

## Decision — backfill the data

A one-shot maintenance script, `scripts/backfill_chunk_citations.py`, rewrites
bare `[^<digits>]` → `[^chunk:<digits>]` in finding bodies across every project
corpus. After it runs, the unchanged reader resolves the citations and the
click→pane path works. **No frontend, validator, or schema change.** No
`SCHEMA_VERSION` bump — this is a data fixup, not a schema change.

### Why backfill, not reader leniency

The rejected alternative was relaxing the readers to also accept bare `[^N]` as a
chunk citation. That was declined because it (a) resurrects the exact ambiguous
grammar #624 deliberately killed, as a permanent shim, and (b) only fixes the web
surface — `finding export/import` and the dangling-citation detection in
`read_finding` would still see bare markers. Backfilling makes the *data* conform
to the one grammar, so every consumer lines up with no dead grammar kept alive.

The no-back-compat rule's usual escape hatch is "re-ingest," but a finding is an
**immutable research artifact** produced by an agent session — it cannot be
regenerated. Backfill is the data-normalization equivalent of re-ingest for data
that can't be re-derived.

### Why a `scripts/` one-shot, not a `bartleby` verb

This is corpus surgery run once per machine to clear a closed set of legacy
findings (new findings can't acquire bare markers — the validators reject them).
A standing `bartleby <verb>` would add permanent surface for a one-time fixup, so
it lives in `scripts/` alongside `release.py` instead.

## Safety properties

- **Lossless.** Pre-#624 a bare `[^N]` was unambiguously a chunk citation (the
  `finding_citations` rows are the source of truth for *which* chunk); a marker
  whose id is no longer a live citation simply renders as the reader's existing
  "no longer available" note.
- **Idempotent.** The regex matches only `[^` + digits; `[^chunk:N]`/`[^url:…]`/
  `[^doc:…]` never match, so re-running changes nothing.
- **Atomic + recoverable.** Default dry-run; `--write` applies under one
  transaction per corpus and first snapshots the DB to
  `<name>.pre-chunk-backfill.bak` via SQLite's online-backup API (WAL-faithful,
  consistent single file), never overwriting an existing backup.
