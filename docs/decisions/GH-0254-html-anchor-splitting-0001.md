# Split TOC-anchored EDGAR/iXBRL HTML into a container + section documents (issue #254)

> Source: [#254](https://github.com/jswest/bartleby/issues/254)

Large SEC filings (S-1s, 10-Ks — hundreds of pages) ingest as one monolithic
`documents` row. That makes the agent slow: the single summary truncates past
`max_summarize_tokens`, and `read_document` on a 300-page doc is a token bomb.
The fix (the issue's adopted "Option C1") is to make the *unit* smaller — split a
filing at its internal table-of-contents anchors into many independently
chunked, embedded, and summarized section documents.

## What splits, and what does not

Splitting is **EDGAR/sec2md only**, and only when a real TOC is present:

- The split lives in `bartleby/ingest/sec2md.py::convert_sections`. It parses the
  raw HTML with BeautifulSoup (already available transitively wherever the
  `sec2md` extra is installed — imported lazily, never at module load), collects
  the `<a href="#id">` links whose targets actually exist in the body, slices the
  body at the top-level ancestors of each target, and converts each slice
  independently via the existing `convert_bytes`.
- It splits **only** when at least `_MIN_SECTIONS_TO_SPLIT` (2) anchors resolve to
  in-document targets *and* at least two slices produce real chunks. A filing with
  no TOC, a dangling/single TOC, or decorative-only anchors returns `[]` and
  ingests whole — byte-for-byte the prior behavior.
- **docling (general HTML) is untouched** — it ingests whole, as before. Only the
  sec2md branch (`html_converter=sec2md` + iXBRL sniff) can split.

## The data model (additive, held at schema 8)

Four nullable columns on `documents` (`schema.py`): `parent_document_id`
(self-FK), `anchor_id`, `section_title`, `section_order`. `NULL` across all four
is the truthful "ordinary whole-file document." A split filing becomes:

- one **container** row — original `file_hash` (`sha256(file_bytes)`), the four
  columns NULL, and **zero chunks of its own**. It is the provenance anchor.
- N **section** rows — each `parent_document_id` → container, carrying the TOC
  anchor id, the link text as `section_title` (free semantic labelling), and its
  TOC `section_order`. Each is a full document: its own chunks, embeddings, and
  (via the normal summarize pass) its own untruncated summary. A section's
  `file_hash` is the derived `sha256(file_bytes + anchor_id)` — keeps `file_hash`
  UNIQUE and re-ingest-stable, and cannot collide with the container's plain
  byte-hash or with a standalone copy of the section's own bytes.

**Held at schema 8.** Per the v0.9.0 omnibus protocol, `SCHEMA_VERSION` stays 8
and these ALTERs are *appended* to the existing `_upgrade_v8_to_v9` step that
#114 created (one consolidated v8→v9 bump ships at assembly). The columns are
nullable with no default, so the self-referential FK `ADD COLUMN` is legal on a
populated table; an existing v0.8.x corpus upgrades in place with every document
reading as a correct-but-unsplit whole. **Sectioning old monoliths is a voluntary
re-ingest, not forced** — the upgrade never invents parents.

## Atomicity / resume

The whole split is one `persist_parse` transaction. Section rows persist
**first**; the zero-chunk container row persists **last**, with the section→
parent links wired only after the container exists. Resume keys on the
container's `file_hash`, so a crash mid-split commits nothing and the file
re-parses cleanly — there is never a window where section rows are visible
without their container, nor where a half-split file reads as complete. This
leans directly on #358's pinned "atomic parse" invariant, now exercised for the
N+1-rows-per-file case (`tests/test_ingest_edgar.py::
test_persist_parse_split_is_atomic`).

## Summaries and `describe_corpus`

A zero-chunk container owes no summary: `documents_needing_summary` already
excludes chunkless documents (the #80 guard), so the container is never handed to
the model, and completeness treats it as done. Sections flow through the normal
summarize pass and each gets a faithful, untruncated summary. The one place the
container would have skewed a number is `describe_corpus`'s unsummarized tally
(`document_count - summarized`): containers are now subtracted out (a container is
any row referenced by another row's `parent_document_id`), so a fully-summarized
split filing no longer reads as forever-incomplete. A container roll-up summary is
an explicit v1 non-goal.

## Why split the raw HTML rather than sec2md's section API

sec2md *does* ship `extract_sections`, but it is item-based and requires a known
filing type (10-K/10-Q/8-K/…). The issue's motivating example is an S-1, which
that API does not cover, and the design is explicitly *anchor/TOC*-driven. Slicing
the raw HTML at TOC-anchor targets is filing-type-agnostic, uses the link text for
free section titles, and reuses `convert_bytes` unchanged per slice — so every
byte of content lands in exactly one section with nothing duplicated or dropped.
