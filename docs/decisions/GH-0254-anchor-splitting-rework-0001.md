# Anchor-splitting rework: preamble section + document-order, TOC-only boundaries (issue #254)

> Source: [#254](https://github.com/jswest/bartleby/issues/254) (critic-loop rework under omnibus [#363](https://github.com/jswest/bartleby/issues/363))

The original #254 split (see
[`GH-0254-html-anchor-splitting-0001.md`](./GH-0254-html-anchor-splitting-0001.md))
shipped three confirmed production bugs in `bartleby/ingest/sec2md.py`. Its own
stated invariant — "every byte of content lands in exactly one section, nothing
dropped" — did not hold. This rework restores it.

## Bug 1 — pre-TOC content was silently dropped

Slicing started at the first anchor target's top-level ancestor, so everything in
`<body>` *before* it — the EDGAR cover page (registrant name, CIK, period,
ticker, shares outstanding) — landed in no section. The zero-chunk container owns
no chunks, so that content vanished from FTS and embeddings entirely.

**Fix.** `_convert_sections_bytes` now emits a synthetic **preamble** section for
body content preceding the first TOC target, when that content is non-empty:

- `anchor_id = "__preamble__"` (`_PREAMBLE_ANCHOR_ID`), `section_title =
  "Preamble"`, `section_order = 0`. The leading/trailing underscores keep the id
  from colliding with a genuine HTML id a filing uses as a TOC target; the
  derived `file_hash = sha256(file_bytes + "__preamble__")` is UNIQUE and
  re-ingest-stable like any section.
- It is an ordinary section: its own chunks, its own summary, indexed in FTS and
  sqlite-vec. Cover-page facts are searchable again.
- The TOC nav block itself is **not** re-indexed as preamble. The TOC's `<a>`
  elements' top-level ancestors are excised from the preamble slice
  (`_slice_between(..., skip=toc_blocks)`), so navigation links don't masquerade
  as front matter.
- When there is no real pre-TOC content (the first target is the first body
  block), no preamble section is emitted — the prior shape is preserved.

The remaining sections renumber after the preamble (`order` is assigned in
emission order), so `section_order` stays a dense 0..N sequence.

## Bug 2 — out-of-order anchors duplicated content; harvesting over-collected

Targets were ordered by **link** occurrence, not **document** position, and
`_resolve_toc_targets` accepted **every** `<a href="#…">` in the filing.

- **Out-of-order duplication.** If TOC link order ≠ document order, a slice whose
  next boundary lay *earlier* in the document ran to end-of-body, so the same
  paragraphs were chunked/embedded/indexed in two sections.
- **Over-harvesting.** In-text cross-references ("see Item 1A"), footnote-return
  markers, and "back to top" links were all treated as TOC entries, producing
  spurious and non-monotone sections.

**Fix — the boundary/harvesting rule.** `_resolve_toc_targets` now:

1. Computes each element's document position from `body.descendants` (a single
   pass).
2. Walks `<a href="#id">` links **in link order**, resolving each to a body
   target. Keeps only **forward** links — the link sits earlier in the document
   than its target (`link_pos < target_pos`). A "back to top" / footnote-return
   link points backward and is dropped. Deduplicates by `anchor_id`, first link
   wins — so an in-text cross-reference to an *already-listed* section is dropped
   as a duplicate.
3. Takes the **longest contiguous run** of the surviving links (contiguity in
   original `<a>` index). A dropped link between two survivors breaks the run, so
   an in-text cross-reference to an *un*-listed section — which sits in the body,
   separated from the TOC by other links — falls outside the run. The longest run
   is the table of contents; ties keep the earliest (the TOC sits at the top of
   the filing).
4. **Sorts the run by document position** before returning.

Sorting by document position is what makes link order irrelevant: every slice now
ends at the next target **in document order**, and only the genuinely last target
runs to end-of-body. Content can no longer be duplicated, and only the contiguous
TOC cluster creates sections.

`_slice_between` gained a `skip` set (top-level blocks to omit) so the preamble
can exclude the TOC nav block; everything else is unchanged.

## Bug 3 — `describe_corpus` container double-subtract

`describe_corpus.py` computed `unsummarized = document_count - summarized -
container_count`, where `container_count` was *every* container. If a container
ever acquired a summary row (an agent runs `save_summary` on one), it was counted
in **both** `summarized` and `container_count`, undercounting `unsummarized`
(possibly negative).

**Fix.** `container_count` now counts only containers **without** a summary row
(`AND document_id NOT IN (SELECT document_id FROM summaries)`). A summarized
container is already accounted for by `summarized`; an unsummarized container is
subtracted exactly once. The tally can no longer go negative.

## Unchanged invariants

- Container persists **last** in one `persist_parse` transaction; section→parent
  links wired only after the container row exists. Atomicity / resume keyed on
  the container's `file_hash` is intact — the preamble is just one more section
  row in that same unit.
- All chunk writes go through `bartleby.db.chunks` typed helpers.
- EDGAR/sec2md only; docling is untouched.
- No schema change — held at SCHEMA_VERSION 9, no new columns. The preamble reuses
  the existing `anchor_id` / `section_title` / `section_order` columns.
