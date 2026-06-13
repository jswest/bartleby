# `search` gains date filters with scan parity, by reusing the shared scope resolver (issue #537)

> Source: [#537](https://github.com/jswest/bartleby/issues/537)

`scan` could filter and sort by `authored_date`; `search` only *surfaced* the
date in its output and could not filter on it. The `search` response already
echoed stubbed `authored_after` / `authored_before` / `include_nulls` /
`excluded_null_dated` keys (via `Scope.echo_into`) — so the contract was
half-built: the keys were there, but no flag could ever set them to anything but
null/0. This wires them to real behavior.

**Reuse the existing shared resolver, don't reimplement.** The date-bound logic
already lives in one place — `resolve_scope` (and its `_apply_date_bound` helper)
in `bartleby/skill_scripts/_tags.py`, which `scan` / `list_documents` /
`describe_corpus` all route through. It already accepts `authored_after` /
`authored_before` / `include_nulls`, validates the bounds (`INVALID_DATE`),
narrows the document set, and computes `excluded_null_dated`. So "extract scan's
date-bound logic into a shared helper" was already done in the #536 wave; this
sub-issue's job was to make `search` a *fourth consumer* of it, not to extract
anything new. The change is therefore deliberately tiny:

- `search.parse_args` calls the shared `add_date_filter_args(p)` (the same
  flag-and-help trio `scan` uses) instead of hand-rolling three `add_argument`s,
  so the CLI surface can't drift from scan's wording.
- `search.work` threads `authored_after` / `authored_before` / `include_nulls`
  into its existing `resolve_scope(...)` call.

That's it for behavior. The `excluded_null_dated` count needed no new wiring:
`scope.echo_into` (already on every search response) emits the true count the
resolver computed, so the previously-stubbed key now carries real data for free.

**Identical semantics to scan, by construction.** Because both scripts call the
same resolver, the bounds are inclusive `YYYY-MM-DD` (`>= after` / `<= before`),
undated documents are excluded by default and counted in `excluded_null_dated`,
and `--include-nulls` keeps them and zeroes the count — with no second copy of
the SQL to drift. A `test_search_date_filter_matches_scan_on_same_bounds` parity
test pins this: search and scan resolve the same surviving document set and the
same `excluded_null_dated` for the same corpus and bounds.

**The date narrows search's document scope, intersecting like every other
filter.** `search`'s `restrict = scope.document_ids` already drives
`_build_scope`, so a date bound that narrows `document_ids` naturally restricts
the document leg, restricts summaries/images to the surviving documents (the
existing `--in-documents` intersection semantics), and drops findings (they have
no document anchor and so no date) — consistent with how `--tag` / `--file-like`
already behave. No new scope branching was needed.

**Schema-stable: no `SCHEMA_VERSION` bump, no re-ingest.** This is a read-side
filter over an existing column; no DDL.

**Out of scope** (per the issue): ranking changes — the date is a filter, not an
RRF signal; and `read_document` date surfacing, which landed in #536.

Touches: `bartleby/skill_scripts/search.py` (the `add_date_filter_args` call,
three kwargs into `resolve_scope`, and a docstring update); a date-filter test
block in `tests/test_skill_search.py` reusing the shared `dated_corpus` fixture
(including the scan-parity test). No change to the shared resolver in `_tags.py`
or to `scan`'s external behavior.
