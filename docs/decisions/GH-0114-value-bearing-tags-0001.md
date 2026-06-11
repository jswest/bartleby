# Value-bearing tags — per-document values via regex extraction (issue #114)

> Source: [#114](https://github.com/jswest/bartleby/issues/114), assembled under omnibus [#363](https://github.com/jswest/bartleby/issues/363)

A tag can now optionally carry a **per-document value**, not just membership. A boolean tag answers "which documents are X?"; a value-tag answers "what is X's value in each document?" — a number/string/date pulled from each document's text by a deterministic regex. It rides the existing tag vocabulary and lifecycle: tags-with-an-optional-value-payload, not a new subsystem.

## Data model (purely additive)

Four nullable columns, NULL-truthful on pre-upgrade rows (a plain boolean tag leaves all four NULL and is unchanged):

- `tags.value_type TEXT CHECK (value_type IN ('number','string','date'))` — the discriminator (NULL = boolean tag) and the cast.
- `tags.pattern TEXT` — the extraction regex (with a `(?P<value>…)` named group).
- `document_tags.value TEXT` — the extracted value, stored canonically as text.
- `document_tags.chunk_id INTEGER REFERENCES chunks(chunk_id)` — the chunk the value was matched in; the citation anchor (a finding cites this chunk, preserving "reached through a chunk, never cited directly"). The unchanged `PRIMARY KEY (document_id, tag_id)` makes "one value per (tag, document)" structural.

## Held-at-8 handling

Per the omnibus protocol, `SCHEMA_VERSION` stays **8** on this sub-PR. The columns land in `db/schema.py` (fresh DBs) AND in a `_upgrade_v8_to_v9` fragment keyed at `8` in `db/upgrades.py` (existing DBs). That step is **dormant** while `SCHEMA_VERSION == 8` — the upgrade loop walks `v < 8` and never reaches it. #254 will append its own ALTERs to the *same* `_upgrade_v8_to_v9` function so the v9 bump ships as one consolidated step; the v0.9.0 assembly commit bumps to 9 and removes the xfail. The v8→v9 ALTERs are written to apply cleanly to released v0.8.x DBs (which already carry ingests/ingest_run_id), so the v5→v6 chain step is left untouched. Because the chain-walk equivalence gate strips back to v4 and recreates `tags`/`document_tags` via that older step, it cannot reconcile the new columns until v9 — so `test_upgrade_chain_walks_from_v4_through_current` is marked `xfail(strict=False)` until assembly.

## Extraction (`extract` skill script)

A new `extract --tag <name> --chunks <id,id,…>` applies the value-tag's stored regex to an **explicit, agent-supplied** chunk-id set (same `comma_int_list` parser as `read_chunks`/`assign_tag`). No implicit corpus-wide sweep — the agent has already located the chunks (via `search`/`scan`) and passes them; many ids in one boot amortize interpreter startup. Semantics:

- **re2, not stock `re`.** Agent-authored patterns run across many chunks are a ReDoS vector and `re` has no timeout; `google-re2` makes catastrophic backtracking impossible by construction. We require a `(?P<value>…)` named group at pattern-compile time (in `add_tag` and `extract`) so the captured span is unambiguous.
- **Conflicts.** If two passed chunks of the *same* document capture *distinct* values, that's an ambiguity → reported in `conflicts`, **neither stored** (never silently pick). Two chunks yielding the *same* value are not a conflict; the first chunk is the stored anchor.
- **No match → `no_match`, never fabricated.** Non-matching chunks (and non-`document`-kind chunks, which have no owning document to anchor) are skipped honestly; the rest of the batch proceeds. A capture that can't satisfy the cast lands in `cast_errors`.
- **Numeric normalization.** `value_type='number'` strips `$ , %` and reads `(…)` as negative before the cast, so a pattern need only *isolate* the digits, not clean them.
- **Re-runnable / overwrite.** `upsert_value` uses `ON CONFLICT (document_id, tag_id) DO UPDATE`, so a later unambiguous extraction replaces the prior value + anchor.

## Creation, manual override, surfacing

- **`add_tag --value-type/--pattern`** (both or neither, else `INCOMPLETE_VALUE_TAG`) creates a value-tag; the pattern is compiled/validated before any write. Name/similarity conflict checks are unchanged.
- **Manual value EXTENDS `assign_tag`** (smallest fix, not a new `set_value` script): `--value [--chunk]` records an out-of-band value, cast per `value_type`, on a value-tag. Rejected on a boolean tag (`NOT_A_VALUE_TAG`); `--chunk` requires `--value`.
- **`read_tags`** reports `value_type`/`pattern` and gains `--boolean-only` (`value_type IS NULL`) — the natural default for filtering contexts, since a value-tag is a field, not a browse category.
- **`list_documents --tag <value-tag>`** attaches a per-document `tag_values` chip (`{value, chunk_id}`); omitted when no value-tag is filtered.
- **`merge_tags` collision rule:** value-preserving on the target. The source's value rides along for documents the target lacks; on overlap the **target's value is kept** and the source's is reported in `value_collisions` (`{document_id, kept, dropped}`), never silently lost.

## Scope parked

JSON/skill surfaces only; web tag chips are parked (`bartleby/web/**` untouched). Out of scope per the issue: structured/tabular extraction (use a structured read + `jq`), LLM-based extraction (the method is a deterministic regex), and cross-document roll-up tables.
