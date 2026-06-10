# Test pins persist_parse's mid-unit rollback (the "atomic parse" invariant) (issue #358)

> Source: [#358](https://github.com/jswest/bartleby/issues/358)

The "atomic parse" invariant — a `documents` row implies a *finished* parse, so
resume logic only re-runs units that left no row — rides entirely on
`Writer.persist_parse`'s single `with self.conn` transaction (`writer.py`). The
suite had no test that forced a failure *after* the `documents` INSERT and
asserted the row rolled back, so a refactor that split that one transaction (or
moved the chunk write outside it) would pass the whole suite while silently
committing a partial parse that reads as complete. This becomes more load-bearing
under #254, which persists N+1 `documents` rows per file — a partially-split file
must not commit a subset.

**This is tests-only — `persist_parse` is already atomic and the test passes on
current HEAD.** Per the issue's TEST-ONLY constraint, the test asserts the
existing guarantee; no production code changed and `SCHEMA_VERSION` is untouched.
Had the assertion failed it would have been a real write-path bug to PARK and
report, not to patch here. It did not fail.

**The failure is injected at the natural validation chokepoint, after the
`documents` row is already in.** `persist_parse` does, in order, inside one
`with self.conn`: clear the prior failure, INSERT the `documents` row, then call
`insert_document_chunks(...)`. That helper's `_validate` pass (`db/chunks.py`)
runs *before* it opens its own inner `with conn`, and rejects an embedding whose
length is not `EMBEDDING_DIM`, raising `ValueError("embedding has N dims, ...")`.
The test's second document chunk carries an embedding of length `EMBEDDING_DIM +
1`, so:

- the `documents` INSERT has already executed (the row exists *within* the
  transaction),
- the first chunk is well-formed, so this is a genuine mid-unit failure,
- the `ValueError` propagates out of the outer `with self.conn`, which rolls the
  whole unit back (APSW commits/rolls back the outermost context on exit).

The test then asserts `COUNT(*) == 0` in `documents`, `chunks`, and `images` —
the three tables `persist_parse` writes — proving nothing partial committed. It
reuses the established `tests/test_scribe.py` fixtures/builders (`isolated_project`,
`open_db`, `Writer`, `ParsedDocument`/`ParsedImage`/`ChunkInput`, `_emb`,
`EMBEDDING_DIM`) and mirrors the shape of `test_writer_persist_parse_dedupes_shared_image`.

touches: `tests/test_scribe.py` (one added test). No product code, no schema
change (`SCHEMA_VERSION` unchanged).
