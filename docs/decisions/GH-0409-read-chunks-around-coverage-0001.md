# GH-0409 — read_chunks `--around-chunk` window coverage

**Date:** 2026-06-11
**Issue:** #409 (part of omnibus #413)

## Context

`read_chunks --around-chunk <id>` returns a target chunk plus `--window N`
neighbours on each side, in `chunk_index` order, scoped to the target's
`(source_kind, source_id)`. The only around-mode tests that existed
(`tests/test_skill_read_chunks.py`) exercised the memory wall — a foreign
finding chunk raising `MEMORY_OFF`, and a session's own finding being
readable. The window arithmetic and the not-found path had no coverage.

(The issue body was stale: it asked for tests that "already exist." We filled
the genuine gaps rather than duplicating the memory-wall cases. No
mode-mismatch-rejection test — #406 is a docstring caveat, not enforced
behaviour.)

## Decision

Added six tests covering the around-mode window mechanics against a freshly
seeded 9-chunk document (`_wide_document` helper, built on the existing
`seeded_project` fixture and the typed `insert_document_chunks` helper):

- interior target with the **default window** (asserts `window == 3` and the
  symmetric ±3 neighbourhood);
- an **explicit `--window 1`** (echoes the window, returns ±1);
- **`--window 0`** (target alone);
- **clamping at document start** (target at index 0, no negative indexes);
- **clamping at document end** (target at the last index, no overrun);
- an unknown id raising the canonical **`CHUNK_NOT_FOUND`** code.

Clamping is asserted via observed behaviour (the BETWEEN range simply returns
fewer rows at the edges) — no production code changed; this is test-only.

## Scope

Touches `tests/test_skill_read_chunks.py` only.
