# `finding read` shared-factoring boundary and footnote-numbering semantics (issue #649)

> Source: [#649](https://github.com/jswest/bartleby/issues/649)

## Shared factoring: `_read_finding_for_display` vs. `_read_finding_for_export`

Two read helpers coexist in `bartleby/commands/finding.py` rather than one merged
path. The reason is that they diverge in what they return:

- `_read_finding_for_export` — title / description / body / resolved citations only.
  Export only needs resolved citations; it never needs dangling ids or external
  citations (both pass through to the artifact verbatim as-is).
- `_read_finding_for_display` — adds `session_name / model / created_at` (the
  metadata subtitle), `dangling` chunk ids, and `external` citation list.  `read`
  needs all of these to build the footnote block.

Merging them into one function with flags would add a return-value conditional on
every caller just to carry optional fields; separate helpers are cheaper and honest
about their different shapes.

Both helpers share `_common.{finding_chunk_and_citation_ids, resolve_citations}`
for the citation read path and deliberately skip the memory-wall check
(`assert_findings_accessible`) — `read` and `export` are human CLI operations over
the local corpus, not agent-session reads that need contamination guards.

## Footnote numbering: sequential across chunk + external, first-appearance order

`_render_body_as_markdown` assigns footnote numbers to markers in a single pass,
left-to-right through the body, combining chunk and external citations into one
sequence (`[^1]`, `[^2]`, …).  Alternatives considered:

- **Two separate sequences** (chunk citations 1…M, external M+1…N): rejected
  because the reader sees inline markers in body order; a gap between chunk and
  external numbers would be confusing.
- **Separate sequences per type** (chunks [^1…], externals [^e1…]): not standard
  Markdown footnote syntax; breaks pager renderers like `glow`.

Deduplication: a repeated `[^chunk:N]` or `[^url:ref]` marker gets the same
footnote number on every occurrence — the number is assigned on first appearance
and reused. This matches standard footnote semantics.

## `_citation_label` extraction

`_inert_marker` (export) shared its branching logic with the chunk-rendering path
in `_render_body_as_markdown` (read): `source_kind == "finding"` → title fallback,
else file+page.  That was factored into `_citation_label(citation) -> str` returning
the bare label; callers add their own prefix (`[corpus: …]` vs. `†` inline in the
single-pass callback).  There is no separate `_live_footnote_label` wrapper —
the `†` prefix is applied directly at the call site.
