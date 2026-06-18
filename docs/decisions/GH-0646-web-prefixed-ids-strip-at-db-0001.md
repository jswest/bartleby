# Web app carries prefixed IDs end-to-end; strips to bare int only at the DB boundary

Issue #646 (implementation), triggered by GH-0624.

GH-0624 made every skill script emit type-tagged entity ids (`document:204`,
`finding:12`, `chunk:15837`, `image:7`, `tag:3`, `summary:N`). The SvelteKit
web app was built for bare integer ids and broke nearly everywhere: route URLs
became `/documents/document:204`, `parseIdParam` did `Number("document:204")`
→ NaN → 404, and prefixed strings bound directly into SQL `IN(…)` / `.get()`
queries against bare-int columns returned no rows.

## Decisions

### Prefixed IDs are the app's identity; the bare int is only for SQLite binding

The web app now carries the prefixed `<type>:<int>` string wherever an id
travels — page URLs (`/chunks/chunk:5`), Map keys, values passed between
functions, hrefs built for the UI. The prefix is stripped to a bare integer
**only at the moment the value is bound to a SQLite parameter**. This is the
"strip at the DB doorstep" architecture.

The alternative (a central normalizer that walks skill-output JSON and converts
every prefixed string to an integer before the app sees it) was considered and
rejected. The walk is keyed by field name; `result.query` and tag names are
string fields that happen not to be ids, and a name-based walk would need an
ever-maintained whitelist to avoid touching them. GH-0624 deliberately chose
prefixed strings because the type information is valuable; stripping it before
the app uses it discards that value.

### `bareId` — one small pure helper

`bartleby/web/src/lib/server/ids.js` exports a single `bareId(value)` function.
It accepts a prefixed string, a bare numeric string, or a bare integer; returns
the integer. It recognises only the six canonical types GH-0624 defines (chunk,
document, finding, image, tag, summary); anything else — including citation
markers like `[^chunk:15837]` or unknown prefixes like `gpt:4` — returns NaN.
Tolerating bare ints makes the helper safe to call on DB-sourced values too
(some id lists the app builds internally are already bare ints).

### `parseIdParam` strips the prefix — doorstep for all route and file loaders

`params.js` replaces `Number(raw)` with `bareId(raw)`, making every `[id]`
route loader and `/files/[document_id]` tolerant of prefixed ids in the URL.
An optional second argument `expectedType` enables type-checking: when provided
and the raw value carries a recognised prefix, the prefix must match the
expected type — a `finding:12` id handed to the `documents` route 404s rather
than silently looking up the colliding row. Bare (non-prefixed) values skip
this check so that DB-sourced bare-int URLs keep working.

### Map-keying discipline in `queries.js` batch helpers

The three `queries.js` helpers that receive skill-output (prefixed) ids and
build Maps for their callers all follow the same pattern:

1. Strip each input id to its bare int for the SQL `IN(…)` binding.
2. Build a `bare→original` map from the input list.
3. After the query, re-key each result row by the *original* (possibly-prefixed)
   id, not the bare int the DB column returns.

This means callers doing `tagMap.get(d.id)`, `summaries.get(r.document_id)`, or
`findings.get(h.source_id)` continue to hit correctly because they hold the same
prefixed id they passed in. The DB-sourced values flowing through `getCitations`
and `enrichScanMatches` (already bare ints) pass through `bareId` unmodified, so
those paths are unaffected.

### URLs intentionally carry the prefix

`/chunks/chunk:5`, `/documents/document:204`, `/findings/finding:12` — approved
by the repo owner. The route path already implies the type; the prefix in the
URL is harmless and `parseIdParam` is tolerant in both directions.

### No behavioural `.svelte` changes

href/link construction in Svelte components passes the (prefixed) id straight
into the URL string, which is now the correct behaviour. The finding-body
citation regex in `findings/[id]/+page.svelte` was already updated by GH-0624
to match `[^chunk:N]` and remains unchanged. The only `.svelte` edit is a
comment correction in `ResultCard.svelte` (its chunk link's "`chunk_id` is an
integer" note was stale now that the id arrives as a `chunk:<int>` token — the
markup is unchanged and still safe, the value being constrained skill output).
