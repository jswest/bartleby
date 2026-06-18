# `scan --body-matches` — first-class regex filter mode (issue #653)

> Source: [#653](https://github.com/jswest/bartleby/issues/653)

## Problem

FTS5 phrase queries treat punctuation as a token boundary, so bare tokens like `5376`, `$120,000`, or `H.R. 815` match nothing via `scan "H.R. 815"`. The existing `--extract '/regex/'` can project captures from chunks already located by an FTS query, but it cannot *locate* chunks — it's a projector, not a selector.

## Flag

`--body-matches '/regex/'` — a `/pattern/`-delimited Python `re.search` filter that selects only chunks whose body text matches. No capture group required (it's a selector, not a projector; use `--extract` for captures). Example:

```
scan "Income" --body-matches '/\$[\d,]+/' --extract '/\$([\d,]+)/'
```

## Design choice: Python post-filter, not registered SQL `regexp` function

Two options were considered:

1. **Register a scalar `regexp(pattern, text)` function on the apsw connection** (in `_attach` in `connection.py`) and push the filter into the SQL `WHERE` clause.
2. **Python post-filter** over the FTS5 result set.

Chose **Python post-filter** for these reasons:

- **No connection.py change.** Adding a scalar to `_attach` would run for every connection open (read paths, ingest, the web UI), not just `scan`. The filter is an agent-supplied pattern that only `scan` needs; coupling it to the connection setup leaks scope.
- **Simpler total accounting.** An honest `total` (pre-pagination count of regex-matching chunks) requires counting the filtered set. With a SQL `regexp` in `WHERE`, a second `SELECT COUNT(*)` would be needed. With the Python walk, the list comprehension materializes the full filtered result set once, `len()` gives the total, and a slice gives the page — one pass, no extra query.
- **FTS5 pre-filters first.** The regex fires on FTS5-matched chunks only, not all chunks, so the O(n) cost is proportional to the FTS match set size — the same size that `--count-by '/regex/'` and `--extract` already walk.

## Performance trade-off

O(n) over the FTS5 match set (not the whole corpus). For a broad FTS query on a large corpus this materializes all matching chunk texts in memory before slicing the page. Acceptable for a filter primitive (the issue explicitly notes this); the existing `--extract` and `--count-by '/regex/'` modes have the same profile. The `--body-matches` path does **not** apply the `CAPTURE_DEADLINE_SECONDS` / `CAPTURE_MAX_MATCHES` guards (those guard the regex-*capture* modes against runaway patterns; a `re.search` selector is cheaper — one boolean per chunk, no iteration inside the match). Very broad patterns on very large match sets are the user's footgun to manage.

## Composition

Composes with all scope flags (`--in-documents`, `--tag`, `--file-like`, `--heading-like`, `--authored-after/before`) and with `--extract` (filter to locate, extract to project). The `--extract` path was updated to apply `body_filter` inline in the list comprehension that already materializes all rows, keeping one walk with both the deadline guard and the regex check.

## Output shape

`body_matches` is echoed at envelope top-level (alongside `query` / `match_mode`), not inside the `filters` object — it is a matching mode, not a scope filter. `filters` is only present when a scope filter is also active.
