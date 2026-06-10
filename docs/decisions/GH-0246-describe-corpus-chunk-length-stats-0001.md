# describe_corpus reports chunk-length stats so the agent self-sizes `--preview`

**Issue:** #246 · **Status:** settled

## Context

The snippet-preview default is 240 chars (`search.py BRIEF_PREVIEW_CHARS`,
`scan.py DEFAULT_PREVIEW`). On templated corpora (OGE/ethics filings, EDGAR,
PACER) the useful payload reliably sits *mid-chunk* — `Signed:` dates, covered-
position parentheticals, bill lists — so 240 chars clips it. An agent on such a
corpus only learns to reach for `--preview 600/2000` after the default has bitten
it several times.

## Decision

Give the agent the signal to right-size `--preview` itself: `describe_corpus`
now reports a `chunk_length` object — `{median, p90, max}` chars — computed over
the same ingested-chunk set as `content_mix` (`source_kind IN
('document','image')`) and narrowed by the same scope filters. Median + p90 is
enough to infer the shape ("chunks here run ~800 chars, the 240 default will
clip most — pass `--preview 1000`"); mean alone misleads on the skewed
distributions these corpora produce.

We **do not** raise the 240-char default globally. 240 is a sensible scan-economy
default for heterogeneous corpora; bumping it would tax tokens on every other
corpus to fix one shape. The fix is a *signal*, not a new default — the agent
decides per corpus.

## Notes

- **Additive, non-schema.** A read-time aggregate over existing chunk text; no
  `SCHEMA_VERSION` bump, no re-ingest.
- **Percentiles are nearest-rank, no interpolation** — integer-valued outputs,
  no spurious precision, and the helper documents the method in one place.
- **Computed in Python, not SQL.** SQLite has no percentile function; gathering
  `length(text)` per row and folding in Python is cheaper and far clearer than a
  window-function/self-join percentile in SQL. This is the one `describe_corpus`
  aggregate that gathers per-row rather than folding in SQL — still cheap.
