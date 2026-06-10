# read_chunks: structured error for unknown --document, with a cross-namespace hint (issue #370)

> Source: [#370](https://github.com/jswest/bartleby/issues/370)

`chunk_id` and `document_id` are independent autoincrement sequences over one
integer space, and `read_chunks` accepts both kinds (`--document`, `--chunks`,
`--around-chunk`). The field test surfaced two failure modes at that seam:

1. `--document <chunk_id>` — a chunk id passed where a document id was meant —
   produced no clean structured error tied to the namespace mistake.
2. `--chunks <document_id>` — a document id passed to a chunk lookup — could
   *silently* return a valid-but-unrelated chunk whenever the document id also
   happened to be a live chunk id, which is the root of a corrupted citation.

## What we changed

- **Unknown `--document` returns `{"error", "code": "UNKNOWN_DOCUMENT"}`** with
  a non-zero exit (renamed from the prior `DOCUMENT_NOT_FOUND`, to match the
  issue's contract). No path emits a raw traceback for a bad id — the runner's
  catch-all already prevents leaked tracebacks, and this seam now raises a
  *meaningful* code rather than relying on that backstop.
- **Cross-namespace hint, one direction at a time.** When the unknown
  `--document` id is in fact a live `chunk_id`, the envelope carries a `hint`:
  `"<id> is a chunk_id — did you mean --chunks <id>?"`. Symmetrically, a
  `--chunks` id that lands in `missing` but exists as a `document_id` gets a
  per-id entry in a `hints` map: `"<id> is a document_id — did you mean
  --document <id>?"`. The `hints` map is present only when at least one missing
  id is a live document id.

## Why the hint is deliberately narrow

The hint fires **only** when the id is unknown in the *requested* namespace but
live in the *other* one. We do **not** warn on an id that is valid in both
namespaces: every chunk id below `max(document_id)` collides, so a both-valid
warning would fire constantly and train agents to ignore it — and an id that
legitimately resolves in the requested namespace carries no detectable mistake
to flag. Failure mode 2 is therefore not fully detectable; the hint hardens the
*error* seam, and the prose habit below covers the silent-success case.

## SKILL.md

The `read_chunks` row gains one sentence teaching the verification habit that
actually catches the both-valid case: check that `source_name`/`file_name` on
what comes back matches the document you meant *before* citing it. That is the
only reliable guard when an id is valid in both namespaces.

No schema change — patch-level. Truly disjoint id namespaces (typed prefixes or
non-overlapping ranges) would be `breaking-schema` and is explicitly out of
scope.
