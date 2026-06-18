# Finding-to-finding links as `[^finding:N]` citation markers (issue #654)

> Source: [#654](https://github.com/jswest/bartleby/issues/654)
> Part of omnibus [#656](https://github.com/jswest/bartleby/issues/656)

## Re-scope: marker, not table

The issue originally proposed a `finding_links` table. The maintainer overrode
this to a citation-marker approach: implement finding-to-finding links as a
`[^finding:<int>]` marker type extending the existing `[^chunk:N]` citation
grammar (#624), with front-end rendering.

**No schema change. No `upgrades.py` entry. No re-ingest.**

## Decision: body-marker convention, unidirectional

A `[^finding:N]` marker in a finding body is a unidirectional reference to
another finding. The body text is the storage — exactly the same rationale as
external citations (#563): the marker already persists in `findings.body`, and
adding a table would duplicate it into a row that backs no FK, no cascade, and
no reverse lookup that the body can't also supply. The body is the single source
of truth; derived information (whether the target still exists) is computed on
read.

## How `[^finding:N]` is parsed, validated, stored, and rendered

### Parsing

A new `_FINDING_CITATION_MARKER = re.compile(r"\[\^finding:(\d+)\]")` in
`_common.py` (symmetric with `_CITATION_MARKER` for `[^chunk:N]`). A new
`extract_finding_citations(body)` returns finding_ids in first-appearance order,
deduped — the same shape as `extract_citations`.

The `_EXTERNAL_MARKER` regex already captures `[^finding:N]` (it matches any
alpha scheme); `finding` is now classified as an internal scheme alongside
`chunk` — handled by its own extractor, excluded from external-marker
classification (`_EXTERNAL_SKIP_SCHEMES`).

### Validation at save/edit

`load_finding_body` (the shared write-path in `_common.py`, called by
`save_finding`, `edit_finding`, and `merge_findings`) now calls
`validate_finding_ids_exist(conn, extract_finding_citations(body))` after the
chunk-citation checks. A reference to a non-existent finding raises
`UNKNOWN_FINDING_LINKS` (naming the missing ids in `unknown_finding_ids`), the
cousin of `UNKNOWN_CITATIONS` for chunk refs.

`reject_malformed_internal_citations` now catches `[^finding:<non-digit>]`
(e.g. `[^finding:abc]`) alongside the existing `[^chunk:<non-digit>]` guard —
both require an all-digit ref (`MALFORMED_CITATION`).

`reject_wrong_typed_citations` previously rejected both `document` and `finding`
schemes. It now only rejects `document` — `finding` is a valid marker. Its
`_REJECTED_CITATION_SCHEMES = ("document",)`.

`reject_malformed_external_citations` excludes `finding` from its scope (via
`_EXTERNAL_SKIP_SCHEMES`) so it isn't double-flagged.

### Storage

Nothing new. The `[^finding:N]` text lives in `findings.body` verbatim.
`finding_citations` (the chunk-citation table) is untouched. No FK, no cascade,
no reverse-lookup row.

### Read: `dangling_finding_links`

`read_finding` now emits `dangling_finding_links: ["finding:<id>", ...]` — the
subset of `[^finding:N]` markers whose target finding no longer exists (deleted
after the body was written). Computed at read time by extracting finding_ids from
the body and diffing against a single `SELECT finding_id FROM findings WHERE
finding_id IN (...)` query. The body is left verbatim (the same provenance
rationale as dangling chunk citations).

`_ids._OUTPUT_FIELD_TYPES` maps `"dangling_finding_links"` to `"finding"` so
the ids are prefixed automatically by `format_output_ids`.

### Web rendering

`findings/[id]/+page.svelte` `renderBody` now handles the `finding` scheme: a
`[^finding:N]` marker in the body becomes a `¶N` superscript inline and a
`margin-note--finding` gutter note with a link to `/findings/N`. The `¶` glyph
(pilcrow) was chosen to distinguish finding links from chunk citations (`†`),
gone chunks (`‡`), and external citations (`§`), following the traditional
footnote sequence.

CSS: `cite-ref--finding` and `margin-note--finding` classes added in `app.css`
with an accent color tone — visually distinct from the amber `--source` and
danger `--gone` variants.

No server-side data is needed: the `[^finding:N]` marker carries the id; the
link to `/findings/N` is built entirely client-side in the renderer.

## What is NOT stored

- No `finding_links` table.
- No new column on `findings`.
- No cascade behavior (if a linked finding is deleted, the marker stays in the
  body and `dangling_finding_links` surfaces it on next read).
- No bidirectional index (a reverse lookup "which findings link to finding N"
  would require a table or a body-scan; neither is in scope here).

## Consequences

- Zero migration cost: existing corpora gain the feature with no upgrade.
- The re-scope from table to marker means this is entirely a citation grammar +
  validation + web-render change, touching no schema, no ingest path, no CLI.
- The `¶` gutter note is an internal link (same origin), so no CSP or
  sanitization concern.
- A future need for reverse lookup or cascade ("delete linked markers when
  target is deleted") would justify a table; the body-marker form does not
  foreclose it (a backfill could parse bodies).
