# `scan --count-by` groups by a regex capture — a fold primitive, deliberately not a query engine

> **Superseded in part by #420 (v0.9.4):** when `scan --extract` began sharing
> this capture machinery, the runaway-guard error codes were renamed
> `INVALID_COUNT_BY_REGEX` → `INVALID_CAPTURE_REGEX` and `COUNT_BY_TIMEOUT` →
> `CAPTURE_TIMEOUT` (now emitted by both `--count-by` and `--extract`). See
> `GH-0420-scan-extract-capture-columns-0001.md`.

Issue #248 (part of the v0.8.6 omnibus, #249). `scan` could find *which chunks
match* and `--count-by document` could count *how many documents match*, but
nothing sat between them: pulling the value out of a templated field and folding
the values together. So agents on form-like corpora (OGE/EDGAR/PACER) kept
hand-rolling `scan "Income:" | python3 -c "...regex...; sum"` — an escape hatch
smooth enough that the missing primitive went unnoticed, and brittle enough
(`Income :` with a stray space, a thousands-separator, a line split) that totals
came out subtly wrong.

## Decision

Generalize `--count-by` from the lone literal `document` to also accept a
`/regex/` carrying one capture group: bucket the FTS-matched chunks by the
captured substring and count. It reuses the existing "bucket and count" shape —
the only new idea is "key = captured group instead of `document_id`."

The hard boundary, stated up front so this stays small: **regex-capture + a
simple fold (count), not a query engine.** No joins across documents, no computed
`where` predicates, no "group by X where Y." That line is what keeps this a
narrow extension of `scan` rather than the first step of a creeping
query-language project. It's useful in exactly the places `scan` itself is useful
(templated corpora with a uniform marker) and degrades to noise exactly where
`scan` already does — it runs *with* the grain of an existing tool.

`--extract` (emit the captured values themselves, not just their histogram) was
named in the issue as an explicit follow-on and is **not** built here: it's a real
new output mode with real decisions (coerce-to-number for summing? multiple
matches per chunk? no-match sentinel?). `--count-by` ships and proves demand
first.

## Choices baked in

- **Per match, not per chunk.** A chunk with two matches contributes two to the
  bucket — an honest frequency. Called out because the alternative (one per
  chunk) is a defensible reading the issue flagged.
- **`/regex/` slash-delimited is the trigger; `document` stays a reserved
  keyword.** Anything that is neither (`banana`) is a clear `INVALID_COUNT_BY`
  error, an uncompilable pattern is `INVALID_COUNT_BY_REGEX`, and a capture-less
  regex is `COUNT_BY_NO_CAPTURE` — all `SkillError`s (exit 1, JSON), matching how
  `INVALID_DATE` is handled, not argparse's exit-2.
- **`--sort` does not apply to regex buckets.** A captured value spans documents
  and dates, so chronological ordering is meaningless; buckets always sort
  count-desc then value-asc. Documented rather than errored, since it's a no-op,
  not a conflict.
- **Output mirrors the document histogram** so the two `--count-by` modes read
  alike: `distinct_value_count` headline, `total_match_count` (full, per-match),
  `groups[{value,count}]`, paginated by `--limit`/`--offset`; `--preview` /
  `--brief` stay rejected (they only shape per-chunk matches).

## Runaway guard — proportionate, because the pattern is agent-supplied

The regex comes from the agent driving the skill, not an external attacker, and
it runs in a short-lived read-only `bartleby skill scan` subprocess — so the
blast radius of a pathological pattern is "that one subprocess spins CPU until
the harness kills it," not a security boundary. The guard is sized to that:

- A **between-chunk wall-clock deadline** (`COUNT_BY_TIMEOUT`, exit 1). It is
  *between* chunks on purpose — a single `re.finditer` call is C code that does
  not return to the Python eval loop, so a `SIGALRM`/`signal`-based "timeout"
  would be deferred until the match completes and is **security theater** against
  catastrophic backtracking. The between-chunk check catches the realistic "broad
  pattern × big corpus" runaway; chunks are size-bounded by the chunker, which
  keeps the residual single-chunk-backtracking gap improbable.
- A **match cap** that flags `truncated: true` with partial counts rather than
  dropping silently (the repo's "no silent caps" rule).

A hard-kill `multiprocessing` worker with `terminate()` would close even the
single-chunk gap, but that's ~30 lines of plumbing per scan for a footgun that
the agent-not-attacker threat model and bounded chunk sizes make remote. Declined
as disproportionate.

## Scope

Additive, **non-schema** — operates on chunk text `scan` already returns. No
`SCHEMA_VERSION` bump, no re-ingest. Touches `bartleby/skill_scripts/scan.py`,
the `scan` rows of `bartleby/skill/SKILL.md`, and `tests/test_skill_scan.py`.
