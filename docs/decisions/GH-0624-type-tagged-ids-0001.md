# Agent-facing entity ids are type-tagged at the interface, not in storage

Issues #624 (implementation) and #623 (the superseded prompt-only predecessor).

Every entity id a skill script emits to stdout JSON is now rendered
`<type>:<id>` — `chunk:15837`, `document:204`, `finding:12`, `image:7`, `tag:3`
(and `summary:N` where a chunk's polymorphic `source_id` points at a summaries
row). Symmetrically, every `--*-id`-style flag and the inline citation marker
accept **only** the prefixed form. The five canonical types are `chunk`,
`document`, `finding`, `image`, `tag`; `summary` rides along solely for
`source_id` rendering.

The driver is a real defect: `chunk_id` and `document_id` are independent INTEGER
PKs whose number ranges overlap, so a bare int is ambiguous. In real sessions a
`document_id` was silently mis-stored as a citation (it parsed, pointed at the
wrong row, and nothing complained). Type-tagging makes that confusion
**structurally impossible** rather than a thing the prompt asks the agent to be
careful about.

## Decisions

### Interface layer only — no schema change

This is parse-on-input, format-on-output. **Storage stays integer.**
`finding_citations` rows are still bare ints; no table, column, `upgrades.py`
entry, or `SCHEMA_VERSION` bump. The new `bartleby/skill_scripts/_ids.py` module
owns the whole convention: `format_id` / `format_source_id`, `parse_id`, the
argparse factories `prefixed_int` / `prefixed_int_list`, and `format_output_ids`
(a recursive walk with a **fixed** field→type map applied at each script's
`work()` return). `source_id` is deliberately *absent* from that map — its type
is the row's `source_kind`, unknowable from the key alone — so it is prefixed
explicitly by kind at its emission sites.

### Hard cutover — bare/untyped ids are rejected, not coerced

Per the repo's no-back-compat rule, there is no deprecation window. A bare int on
a flag (or a wrong-typed value, e.g. a `document:` id handed to `--chunks`) fails
argparse → the runner's `USAGE_ERROR` envelope. A wrong-typed flag value never
silently looks up the colliding row.

### `[^chunk:N]` integrates with the existing external-marker grammar

The citation marker moves from bare `[^N]` to type-tagged `[^chunk:N]`. This
reuses the `[^<scheme>:<ref>]` grammar that already powers `[^url:…]` /
`[^doc:…]` external citations (#563): `chunk` is the canonical **internal**
scheme, routed to chunk-citation handling, while `url`/`doc` stay external and
unchanged. The cutover is hard on both edges:

- bare `[^N]` and caret-less `[N]` are rejected (`MALFORMED_CITATION`);
- `[^document:N]` / `[^finding:N]` are rejected with a dedicated
  `WRONG_CITATION_TYPE` ("cite the chunk you were handed, never a document/
  finding id") — the #623 confusion made loud at save time;
- the malformed-external check skips `chunk` (internal) and the rejected schemes
  so they aren't double-flagged.

Storage is unaffected: `extract_citations` still yields bare chunk_ids and
`finding_citations` still stores ints. The web finding view
(`findings/[id]/+page.svelte`) and the `bartleby finding export/import` path
both had `[^N]` regexes updated to `[^chunk:N]` so they keep resolving citations.

## Out of scope (explicitly)

This change makes a *wrongly-typed* citation impossible. It does **not** verify
that a correctly-typed citation — a valid `chunk_id` that simply doesn't support
the claim — is relevant. That provenance/relevance check is a separate future
issue. This PR does not build it and does not claim to fix it.
