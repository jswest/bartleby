# `--file-like` filename filter routed through the shared `Scope` (issue #366)

> Source: [#366](https://github.com/jswest/bartleby/issues/366)

Arbitrary corpora encode load-bearing metadata in `file_name` (dated exports, per-entity files, EDGAR accessions, `<BioguideID>__<date>__<slug>.md`). The `documents.file_name` column was returned by every match but could not be *filtered* on — forcing corpus-wide scans paginated and filtered client-side. This adds `--file-like <pattern>` (SQL `LIKE`, repeatable for OR) to `scan`, `search`, and `list_documents`.

## Where the filter lives

The patterns resolve to a `document_id` set inside the existing `resolve_scope` rather than being special-cased per script. All three scripts already drive entirely off `Scope.document_ids` (scan/list_documents via the `IN (...)` / `restrict_in` predicate, search via `_build_scope` over the resolved `restrict` set), so threading the LIKE through the single shared resolver is the right altitude: one new field on `Scope`, one new intersection step, and every consumer picks it up for free — including the `filters` echo, which already serializes `Scope` uniformly via `echo_into` / `filters_dict`.

`intersect_file_like_filter` mirrors the existing `intersect_tag_filter`: the patterns OR together (`documents_matching_file_like` builds `file_name LIKE ? OR file_name LIKE ? ...`), and the result intersects (ANDs) with the tag/in-documents scope already computed. It runs *after* the tag intersection and *before* the date bound, so `excluded_null_dated` is still counted over the filename-narrowed base. An empty intersection yields `[]`, which the scripts already short-circuit to zero hits.

## Pushdown and safety

The LIKE is pushed down to SQLite as a parameterized predicate — the user's pattern is bound, never interpolated into the SQL text. Resolution is a single `SELECT document_id FROM documents WHERE <ORed LIKEs>` per call.

## Echo and docs

`file_like` is added to the `Scope.filters_dict()` echo (so `{tags, in_documents, file_like, authored_after, ...}` is the uniform shape across scan/search/list_documents/describe_corpus), and `Scope.active` now treats a non-`None` `file_like` as a scope filter so the `filters` object appears whenever it's set. The flag is documented in each script's argparse `--help` (shared `add_file_like_arg` in `_common.py` — its help string escapes `%` as `%%` because argparse `%`-formats help text) and in the matching SKILL.md rows/examples.

No schema change — `documents.file_name` already exists, so `SCHEMA_VERSION` is untouched and no re-ingest is needed. `describe_corpus --file-like` is explicitly out of scope (issue follow-up).
