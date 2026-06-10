# TOC-excision data-loss + double-linked-TOC contiguity fix (issue #254)

> Source: [#254](https://github.com/jswest/bartleby/issues/254) (final critic-loop rework under omnibus [#363](https://github.com/jswest/bartleby/issues/363))

The anchor-splitting rework (see
[`GH-0254-anchor-splitting-rework-0001.md`](./GH-0254-anchor-splitting-rework-0001.md))
restored "every byte of content lands in exactly one section, nothing dropped" —
but the critic's empirical second pass found the invariant still failed in two
edge shapes, plus one regression. This pass fixes all three in
`bartleby/ingest/sec2md.py`. No schema change (held at SCHEMA_VERSION 9).

## Defect 1 — TOC-block excision deleted real content (DATA LOSS)

To keep the TOC nav out of the synthetic preamble, the rework excised the
**top-level ancestor** of each TOC link (`toc_blocks = {id(_top_level_ancestor(el))
…}`, fed to `_slice_between(skip=…)`). A top-level block is a coarse unit, so this
over-excised whenever the nav did not occupy a block of its own:

- **Cover text sharing the TOC's page-div.** A genuine TOC nested inside the same
  top-level page-`<div>` as the cover prose (registrant name, CIK, period) caused
  the whole div — cover text included — to be skipped. The cover facts vanished
  from every section, FTS, and embeddings.
- **Two inline forward links in a content paragraph.** Prose like "…including
  <a>Part II</a> and <a>Part III</a>…" is not a TOC, but the two forward links form
  a run (`_MIN_SECTIONS_TO_SPLIT = 2`). The doc mis-split, and the host paragraph —
  the run's top-level ancestor — was excised, so its prose hit token-count 0 across
  all chunks.

**Fix — excise only a *pure nav container*, and only treat the run as a TOC when it
has one.**

- `_is_pure_nav_block(block, link_ids)` is true only when **every** non-whitespace
  text node in `block` lives inside one of the run's `<a>` links — i.e. the block
  holds nothing but TOC links and whitespace.
- `_enclosing_nav_block(link, body, link_ids)` returns the **outermost** ancestor
  (below `<body>`) that is still pure-nav, or `None`. For a link-list div it returns
  the div; for a TOC table it returns the whole table (cells/rows/table are each
  pure-nav); for an inline prose link it returns `None`.
- **Excision is now a tree edit, not a top-level skip-set.** Before slicing,
  `_convert_sections_bytes` `.extract()`s each link's enclosing nav block from the
  soup. Removing the *nav container* — even one nested below the top level — leaves
  co-resident prose in place: the cover page sharing a div keeps its text (only the
  inner link-list div is pulled), and the slice walk then puts every remaining byte
  into exactly one section. `_slice_between` no longer needs a `skip` set.
- **TOC gate.** `_resolve_toc_targets` now requires at least `_MIN_SECTIONS_TO_SPLIT`
  of the run's links to have a non-`None` enclosing nav block. An inline-prose run
  has none, so it is not a TOC: the file ingests whole and nothing is excised. This
  reuses the same nav detection as the excision, so a run that splits always has a
  nav block to remove, and a run inline in prose neither splits nor excises.

The decision doc's "every byte lands in exactly one section / nothing dropped"
invariant holds again: content that co-resides with TOC links now lands in the
preamble (or its section); only the pure-nav container is removed.

## Defect 2 — double-linked EDGAR TOCs stopped splitting (REGRESSION)

The classic EDGAR TOC table links **both** the item title and the page number to
the same anchor. The rework deduped by `anchor_id` while collecting links
(`if anchor_id in seen: continue`), so every second (page-number) link was skipped.
That left gaps in the surviving links' `link_idx`, and `_longest_contiguous_run`
treats a `link_idx` gap as a run break — so every run collapsed to length 1 → `[]`
→ whole-file ingest. The pre-rework code split these correctly.

**Fix — measure contiguity over kept-or-duplicate links; dedup the target list, not
the run.**

- The link-collection loop keeps **every** forward link, including duplicate links
  to an already-listed anchor. Consecutive item+page links stay adjacent in
  `link_idx`, so the run survives at full length and the table still splits.
- The `seen`/anchor dedup moved **after** the run is chosen: each anchor still
  yields exactly one target (one section per id), and the first link's text wins as
  the title (the item title, not the page number).
- A backward "back to top" / footnote-return link is still excluded (not forward),
  so it still breaks the run; an in-text forward cross-reference to a listed anchor
  is now kept for contiguity but collapses out in the target dedup — it does not
  create a second slice. The out-of-order, cross-reference, and preamble behaviours
  from the prior rework are unchanged.

## Unchanged invariants

- Preamble still captures pre-TOC content; out-of-document-order anchors still don't
  duplicate; in-text cross-reference / back-to-top / footnote-return links still
  don't create spurious sections; targets are still sorted by document position
  before slicing.
- Container persists **last** in one `persist_parse` transaction; all chunk writes
  go through `bartleby.db.chunks` typed helpers; EDGAR/sec2md only, docling
  untouched.
- No schema change — held at SCHEMA_VERSION 9.
