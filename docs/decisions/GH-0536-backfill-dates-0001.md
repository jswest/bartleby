# `scribe backfill-dates` bulk-sets authored_date from a filename, with honest stub summaries (issue #536)

> Source: [#536](https://github.com/jswest/bartleby/issues/536)

`authored_date` lives on the `summaries` table and was populated **only** by the
LLM summarizer at ingest. A templated corpus ingested without summarization (the
press-release corpus: `summary_coverage` 0/174,787) therefore had *zero* dates,
so `scan --sort date` and `--authored-after/before` were dead on arrival even
though every filename encodes a date. `save_date` sets one document at a time and
needs a pre-existing summary row — useless at this scale.

**A human-run CLI admin op, deliberately off the agent surface.**
`bartleby scribe backfill-dates <project> --from-filename '(?P<date>…)'
[--match-path] [--overwrite] [--dry-run]` is a sibling of `scribe` ingest and
`project upgrade`, not a skill script — it's a bulk write over a whole corpus,
the kind of thing an operator runs once, not something an agent reaches for
mid-research. It lives in `bartleby/commands/backfill.py` and is wired as an
*optional* subcommand under `scribe` (bare `bartleby scribe --files …` is
unchanged; `--files` is validated in dispatch, not via argparse `required=True`,
so a subcommand invocation doesn't trip it).

**Schema-stable: no `SCHEMA_VERSION` bump, no re-ingest.** The `authored_date`
and `model` columns already exist; the command only INSERTs/UPDATEs `summaries`
rows and the read side only adjusts queries. No DDL, so the release drift gate is
not implicated.

**The stub contract.** A document with an existing summary gets an
`UPDATE summaries SET authored_date` (only where it `IS NULL`, unless
`--overwrite`). A document with *no* summary gets a date-only **stub**:
`model='backfill'`, empty `title`/`description`/`text`, the date — and **no
summary chunks**. Empty text is not chunked or embedded, so the typed-chunk
helpers are correctly never called here (the chunks discipline holds: the stub
simply has nothing to chunk). The sentinel is defined once as
`BACKFILL_MODEL = "backfill"` in `bartleby/lib/consts.py`, imported wherever the
read side discriminates so a rename can't drift copy-to-copy.

**The read side must not let a stub lie.** A raw `COUNT(*) FROM summaries` would
let stubs inflate coverage, and a `''` title/text would read as a (blank)
summary. So:

- `describe_corpus.summary_coverage.summarized` counts `WHERE model != 'backfill'`
  — a stub correctly reads as *unsummarized*. (`date_coverage` /
  `documents_by_year` are left alone: a stub's date is real, so it *should* count
  as dated.)
- `list_documents.has_summary` is `(summary_id IS NOT NULL AND model != 'backfill')`
  — false for a stub — and the stub's `title`/`description` are suppressed to
  NULL, while `authored_date` still rides along from the row.
- `read_document --summary` returns `null` (not `''`) for a stub, and the
  envelope gains a new `authored_date` field populated regardless of stub status
  (the date is real even when the summary text is suppressed).

**Sections (#254) get the parent's date, via a stub.** Section rows share their
parent's `file_name` (`bartleby/ingest/writer.py`), so the same regex matches
them and they inherit the parent's date — which is what we want, so chunk- and
section-level date filtering works. We create stubs for section rows too rather
than special-casing them out.

**Safety, by construction.** The regex *must* carry a named `date` group
(refused up front via `FilenameDateError` before any document is scanned — an
operator mistake, not a per-doc miss). A per-document match is normalized through
the existing `normalize_authored_date`; a match that captures a non-date
(`2024-13-40`) is **counted and reported as invalid, never silently written as
NULL**. `--dry-run` mutates nothing and prints the same counts (matched /
unmatched / would-insert-stub / would-update / invalid) plus a few sample
`file_name → date` pairs — essential before a 174k-doc run. The whole thing is
idempotent: a second run re-fills nothing (NULL-only by default) and
re-INSERTs nothing (the stub already exists). The filename-extraction helpers
(`compile_filename_date_regex` / `extract_filename_date`) are colocated beside
`normalize_authored_date` in `bartleby/ingest/summarize.py` so a later
metadata-facet sub-issue (#48 seeds) can reuse the named-capture mechanism.

**Out of scope** (explicitly not built here): high-cardinality metadata facets,
`--where`, frontmatter parsing, the metadata sidecar.

Touches: new `bartleby/commands/backfill.py`; `bartleby/ingest/summarize.py`
(extraction helpers + `FilenameDateError`); `bartleby/lib/consts.py`
(`BACKFILL_MODEL`); `bartleby/cli.py` (the `scribe backfill-dates` subparser +
dispatch); the three read-side scripts `read_document.py` / `list_documents.py` /
`describe_corpus.py`; a reusable `dated_corpus` fixture in
`tests/_skill_fixtures.py`; and `tests/test_backfill_dates.py`.
