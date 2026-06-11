# `scan` + `search` restrict the FTS `MATCH` to the `text` column (issue #464)

> Source: [#464](https://github.com/jswest/bartleby/issues/464)

`chunks_fts` indexes both `text` and `section_heading`, so the bare
`chunks_fts MATCH (<expr>)` on both surfaces also fired on heading-only hits — a
chunk whose snippet never contains the query term. On `scan` this inflated
grep-totals (and `--count-by` histograms) with rows whose `text` doesn't hold
the term; on `search`'s full-text leg it injected heading-only chunks into the
RRF fusion. The two surfaces disagreed on what "matches" means, and neither
agreed with the snippet the agent actually reads.

Both `MATCH` sites now column-qualify the FTS expression to the body text using
FTS5's column-filter syntax — `{text} : (<expr>)` — wrapped **only when the
expression is non-empty** (an empty token set must stay `""`, never a malformed
`{text} : ()`). The wrap lives in one shared helper,
`bartleby.skill_scripts._common.text_qualified_fts`, applied at the param site in
`scan._scan` and `search._fts_search` *after* each surface's existing
empty/no-match short-circuit, so that logic is untouched. This is **query-level
only**: no schema bump, no re-index — `chunks_fts` still indexes both columns;
we simply stop asking it to match the heading one.

**Cross-surface parity was the deciding call (2026-06-11).** The prior behavior
matched *both* columns on both surfaces; we deliberately moved both to text-only
together rather than fixing one and leaving the other ambiguous. Deliberate
heading recall does not disappear — it relocates to the tools built for it:
`scan`'s `--heading-like` (#367, a parameterized `section_heading LIKE`
predicate) for structural filtering, and `search`'s semantic (vector) leg, which
embeds the chunk text and is unaffected by the FTS column restriction. Docstrings
on both scripts and the `search`/`scan` rows in `SKILL.md` state the text-only
contract and point at those escape hatches. Inverse tests assert a heading-only
term (`scan`: a chunk with `Appendix` only in its heading; `search`: the
`Methods` heading in the shared fixture) yields no `scan` match and no
`search` FTS-leg hit, with positive controls that real body terms still match and
that `--heading-like` still reaches the heading-only chunk.
