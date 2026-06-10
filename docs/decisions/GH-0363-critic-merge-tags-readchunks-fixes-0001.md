# Critic-loop fixes: merge_tags value safety + read_chunks walled-chunk hint (omnibus #363)

> Source: [#363](https://github.com/jswest/bartleby/issues/363) (omnibus v0.9.0 critic loop)

Three confirmed 🟡 correctness findings surfaced by the `/ultraship` critic over
#114's `merge_tags` and #370's `read_chunks`. All patch-level, no schema change,
no `SCHEMA_VERSION` bump.

## A — merge_tags silently destroyed a source value (data loss)

The value-collision report required *both* sides non-NULL
(`d.value IS NOT NULL AND s.value IS NOT NULL`). When the **destination** held a
plain assignment (value NULL — someone `assign_tag`'d the value-tag without
extracting a value) and the **source** held an extracted value for the same
document, the `INSERT OR IGNORE` kept the destination's NULL row, then the source
row was deleted with the source tag — value + chunk anchor permanently lost, and
*not* reported in `value_collisions`. The docstring promised values are never
silently lost.

**Fix (carry, don't drop):** before the `INSERT OR IGNORE`, an `UPDATE` carries
the source's `value`/`chunk_id` onto any destination row that is a plain
assignment (`value IS NULL`) where the source holds a value. No silent loss; this
is not a kept/dropped collision (the destination had no competing value), so
`value_collisions` stays empty for these. The chosen rule — *carry the value* —
matches the docstring's "never silently lost" promise and is now documented
there. The collision path (both sides hold a *value*) is unchanged: target's
value kept, source's reported as `dropped`.

## B — merge_tags allowed a mixed-type merge (values become invisible)

Merging a value-tag into a boolean tag (target `value_type IS NULL`) copied
`value`/`chunk_id` onto a tag whose value-read paths all gate on
`value_type IS NOT NULL` (`list_documents._value_tag_values`, `read_tags`
display), so the carried values were unreachable — and the source's
`value_type`/`pattern` were deleted with the source tag.

**Fix (refuse before mutating):** `merge_tags` now rejects any mixed-kind merge
(value-tag → boolean tag *or* boolean tag → value-tag) with
`{"error", "code": "MIXED_TYPE_MERGE"}` and a non-zero exit, computed from
`TagRow.is_value_tag` immediately after both tags resolve — before any write.
Same-kind merges (two value-tags or two boolean tags) are unaffected.

## C — read_chunks cross-namespace hint over-fired on walled finding chunks

The hint "`<id>` is a document_id — did you mean --document `<id>`?" fired for any
`missing` id that is a live `document_id`. The old comment claimed memory-walled
foreign finding chunks "aren't document_ids, so they never trigger a hint" — this
was **false**: a walled finding chunk's id lands in `missing`, and chunk/document
ids overlap freely, so a walled chunk id colliding with a real document_id
mis-fired the hint, telling the agent to `--document N` for an id that is really a
(walled) chunk.

**Fix:** (1) the comment is corrected; (2) the hint is suppressed for any
`missing` id that still EXISTS as a `chunk_id` (reusing the existing
`_live_chunk_ids` helper) — the id is genuinely a chunk in this corpus, just
walled. The hint now fires only when the id is *not* a chunk here at all, which
preserves the director's binding rule (never hint on an id valid in the requested
namespace).

## Tests

- `test_merge_value_tags_carries_value_onto_null_destination` — NULL destination
  absorbs the source value + anchor; no entry in `value_collisions`.
- `test_merge_value_tag_into_boolean_tag_refused` /
  `test_merge_boolean_tag_into_value_tag_refused` — `MIXED_TYPE_MERGE`, non-zero
  exit, nothing mutated.
- `test_read_chunks_no_hint_for_walled_finding_chunk_colliding_with_document` —
  a walled finding chunk whose id is forced to equal a live document_id produces
  NO hint (and still leaks nothing).

Full suite green (821 passed, 0 xfailed, 0 failed).
