# External citations are a body-marker convention, not a stored relational kind

Issue #563 (part of the findings/scan agent-surface-hardening omnibus, #565).

Findings may now carry **external citations** — a web URL or an external-dataset
document ref — *alongside* (never instead of) the mandatory ≥1 corpus-chunk
citation. They live in the finding body as alpha-prefixed footnote markers,
`[^url:<url>]` and `[^doc:<ref>]`, and are parsed and rendered **on read**. No
schema change: no table, no column, no `upgrades.py` entry, no `SCHEMA_VERSION`
bump.

## Decision

External citations are a **body-marker convention**, parsed on read, not a
stored row.

The plan interview deliberately resolved this away from a schema change. The
existing `finding_citations` table earns its keep only for *chunk* citations,
where every column is load-bearing:

- a real foreign key (`chunk_id`) with `ON DELETE CASCADE` to `chunks`,
- ingest-time validation that the cited chunk exists (`validate_chunk_ids_exist`),
- location enrichment on read (`resolve_citations` → file_name / page_number),
- reverse lookup (which findings cite this chunk).

An external ref has **none** of those affordances. It is opaque text: nothing to
foreign-key to, no row to cascade from, no location to enrich, no reverse-lookup
in scope, and — per the live-data guardrail — **nothing is fetched** to validate
it. The only fact we hold is the marker string the agent wrote, and that string
**already persists** in `findings.body`. Adding a table would duplicate the body
text into a row that backs no constraint and answers no query the body can't.

So the marker in the body *is* the storage. The read paths
(`read_finding`, `save_finding`/`edit_finding`/`merge_findings` output, and the
web finding view) compute the external-citation list from the body on every
read via `extract_external_citations` in `_common.py`. The only save-time check
is **well-formedness** (`reject_malformed_external_citations`): the scheme must
be `url` or `doc` and the ref non-blank — no network, no FK, no row.

## Why the ≥1-chunk invariant is untouched

The external marker's scheme is an alpha word (`url` / `doc`). The chunk-citation
extractor (`extract_citations`, the regex `\[\^(\d+)\]`) is digit-only, so it
**ignores** external markers for free. The `NO_INLINE_CITATIONS` rule therefore
keeps counting only true `[^N]` chunk markers: a finding whose body carries only
external markers still has zero chunk citations and is still rejected. External
citations strictly *supplement* the corpus-chunk requirement; they can never
substitute for it. A separate extractor was added rather than touching what
`extract_citations` counts, precisely so this invariant can't drift.

## Consequences

- Zero migration cost: existing corpora gain the feature with no upgrade —
  external markers in any already-saved body render distinctly on next read.
- The body stays the single source of truth for the whole finding (the
  verbatim-echo contract), now including its external attributions.
- A future need for FK-like affordances on external refs (dedup index, reverse
  lookup, fetch-and-cache) would justify revisiting; none is in scope here, and
  the body-marker form does not foreclose it (a backfill could parse bodies).
