# `section_heading` in scan output + a `--heading-like` chunk filter (issue #367)

> Source: [#367](https://github.com/jswest/bartleby/issues/367)

`chunks.section_heading` is queryable and `read_chunks` returns it, but `scan` couldn't *filter* on it and (per the issue's framing) document structure was invisible at scan time. Two severable pieces: surface the heading in default-mode matches, and add a `--heading-like` filter. No schema change — `chunks.section_heading` already exists, so `SCHEMA_VERSION` is untouched and no re-ingest is needed.

## Output piece was already in place

Default-mode `scan` matches have carried `section_heading` since #105 (the `--brief` triage projection landed it in the default object and explicitly dropped it from `--brief`, which stays locators-only). So part 1 of the issue needed no code change — only a SKILL.md note and a regression test asserting presence in default / absence in `--brief`. The `Output:` docstring block was already accurate.

## `--heading-like` is chunk-level, so it does *not* go through `Scope`

`--file-like` (#366) routes through `resolve_scope` because `file_name` is a *document*-level attribute resolvable to a `document_id` set. `section_heading` is a *chunk*-level column — two chunks of the same document carry different headings — so it cannot be folded into `Scope.document_ids`. Threading it through the shared resolver would have been the wrong altitude.

Instead the filter is pushed straight into `scan`'s own `where` / `params`, built once and reused by all three modes (default, `--count-by document`, `--count-by '/regex/'`) since they share that predicate. The patterns OR within the group (`c.section_heading LIKE ? OR ...`) and the group ANDs with everything else (it's appended to the existing `where`). Chunks with a NULL heading never match (SQL `LIKE` on NULL is NULL → falsy), which is the intended "structure is invisible" exclusion.

## Pushdown and safety

The LIKE is parameterized — the agent's pattern is bound, never interpolated into SQL text — matching the `--file-like` discipline.

## Echo

Because `--heading-like` isn't a `Scope` field, it can't ride `Scope.filters_dict()`. The `_envelope` helper folds `heading_like` into the same `filters` object after `Scope.echo_into`, creating the object via `setdefault` when the scope alone was unfiltered. So a bare `scan "x" --heading-like '...'` still surfaces `{"filters": {"heading_like": [...]}}`, and a scoped call gets `heading_like` alongside `tags` / `in_documents` / `file_like` / the date keys. The key is present only when the flag is.

## Scope of the change

Edits are tightly anchored to `scan.py` (one argparse block, one `where` append, one `_envelope` line) so #368 (a later wave touching the same file) rebases cleanly. The `--heading-like` arg is declared inline in `scan.py` rather than as a shared `_common.py` helper — unlike `--file-like` it has a single consumer, and `search`/`list_documents` get no heading filter (per the issue, `search` is an explicit follow-up). Per the issue, `--count-by` gains no heading *bucketing* (`--count-by '/regex/'` over text already covers that, and headings are now in the output for client-side bucketing).
