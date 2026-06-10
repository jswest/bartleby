# `authored_date` in every scan/search match (issue #368)

> Source: [#368](https://github.com/jswest/bartleby/issues/368)

`authored_date` is filterable (`--authored-after`/`--before`) and `list_documents` returns it, but `scan`/`search` matches didn't carry it — so an agent triaging matches by time had to re-look-up each document or parse dates out of filenames (the latter only works on corpora whose filenames happen to encode dates). The fix surfaces `authored_date` (nullable) on every match in both tools, in **all** modes including `--brief`. No schema change — `documents.authored_date` and `summaries.authored_date` already exist, so `SCHEMA_VERSION` is untouched and no re-ingest is needed.

## Source of truth is the summary's `authored_date`, not the document's

The schema has `authored_date` on both `documents` and `summaries`, but the *populated, agent-facing* value is the summarizer-inferred one on `summaries`: `list_documents` selects `s.authored_date`, and the date filters/sorts (`_tags.py`, scan's `_CHUNK_ORDER_BY`/`_DOC_ORDER_BY`) all bound and order on `s.authored_date`. To return the same value the agent already filters and sorts by, scan/search read `s.authored_date` (NULL when there's no summary, or a summary with no inferred date) — not the rarely-populated `documents.authored_date` column. This keeps the field consistent across `list_documents`, the date filters, and now the match objects.

## No extra query — fold into the join the row already needs

The issue's hard constraint is "no second query per match." Each tool already joins the row that carries the date:

- **scan** default-mode already does `LEFT JOIN summaries s` (it's what `--sort date` orders on). Adding `s.authored_date` to that same SELECT costs nothing — no new join, no new query. It then rides into both the default match dict and the `--brief` locator dict.
- **search** resolves per-hit locators through the shared `chunk_locations` helper in `_common.py`, which already does the per-`source_kind` join to the underlying document (document→`documents`, summary→`summaries JOIN documents`, image→`document_images JOIN documents` via `_image_anchors`). Folding `s.authored_date` into those *existing* joins (a `LEFT JOIN summaries` where one wasn't already present) is the no-extra-query path. search just reads `loc["authored_date"]` alongside `file_name`/`page_number`.

## Why `chunk_locations`, not a search-local lookup

`authored_date` is locator-grade — the same shape as `file_name`/`page_number`, which already live in `chunk_locations` and resolve the chunk's underlying document per kind. Putting it there (rather than special-casing it in `search.py`'s result loop) is the right altitude: it's one resolver that already knows how each `source_kind` maps to a document, so document/summary/image hits all get the date and findings fall through to `None` (no document anchor). `read_chunks` also calls `chunk_locations`, but it builds its output dict by explicitly picking keys, so the new `authored_date` key is inert there — the blast radius stays at search, as the issue intends. The alternative (a second per-hit query keyed off resolved document_ids) would have violated the no-extra-query constraint.

## `--brief` carries it (unlike #367's `section_heading`)

Deliberate divergence from #367: `section_heading` is default-only and explicitly dropped from `--brief`, because it's structural context. `authored_date` is a *locator* — you triage and time-order by it — so it stays in the `--brief` projection of both tools. The brief docstrings/help and SKILL.md now say so explicitly.

## Scope of the change

Anchored to the four declared surfaces plus `_common.py`: scan's default-mode SELECT + both match builders; search's `chunk_locations`/`_image_anchors` join and the two result projections; both `Output:` docstring blocks and `--brief` help strings; and the SKILL.md "Reading search results" + scan-row guidance. `_common.py` wasn't in the issue's declared-touches list, but it's the only no-extra-query home for search's cross-kind resolution and the change is additive/inert for its other caller. Tests assert `authored_date` is present in default **and** brief modes and is `null` for an undated document (and `null` for findings).
