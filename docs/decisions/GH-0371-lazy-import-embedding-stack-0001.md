# Lazy-import the embedding stack for read-only skill scripts (issue #371)

> Source: [#371](https://github.com/jswest/bartleby/issues/371)

The FTS-only read scripts (`scan`, `list_documents`, `read_chunks`, `describe_corpus`) paid the embedding/model stack's import cost on every invocation even though they never embed. The cost wasn't in those four scripts — it was in the two shared modules they import: `bartleby/skill_scripts/_common.py` imported the chunker (`chunk_markdown_string`) and `embed_texts` at module top, and `bartleby/skill_scripts/_tags.py` imported `embed_texts` at module top (`_tags.resolve_scope` is what all four read scripts pull in). Importing the chunker dragged in `docling` + `filetype` at module load.

The fix is the smallest one that fits: move those top-level heavy imports to **function-local** imports inside the only functions that embed — `embed_body_chunks` in `_common.py` (the EMBED phase of the finding/summary write path) and `find_similar_tag` in `_tags.py` (the tag-similarity check). In `find_similar_tag` the import sits *after* the empty-vocab early-return and the normalized-name exact-match loop, so an exact-name hit never triggers it. The `db.chunks` typed helpers, `bartleby.config`, and `bartleby.providers` are all light and stay at module top.

**Before/after (importing `bartleby.skill_scripts.scan` in a fresh interpreter):**
- Modules loaded: **428 → 407** (21 fewer), eliminating all 15 `docling.*` / `filetype.*` submodules plus the lazy `bartleby.ingest.embed` loader. `numpy` still rides in via `sqlite-vec` on the DB connection — that's core read-path infrastructure, not the embedding stack, and is out of scope.
- `-X importtime` cumulative for `scan` dropped from ~71.5ms to ~67ms; for `describe_corpus` ~72.6ms → ~67ms. The residual is shared infra (`apsw`, `sqlite_vec`→`numpy`, `pydantic`) that the read path legitimately needs; the structural module-set delta is the honest measure.

A new regression test (`tests/test_skill_read_import_cost.py`) asserts in a *fresh interpreter* (so module caching can't mask a leak) that importing each of the four read scripts does not pull in `bartleby.ingest.embed`, `bartleby.ingest.chunk`, `sentence_transformers`, `torch`, `docling`, or `filetype`. Test monkeypatch targets that previously patched `_common.embed_texts` / `_common.chunk_markdown_string` / `_tags.embed_texts` were repointed at the source modules (`bartleby.ingest.embed.embed_texts`, `bartleby.ingest.chunk.chunk_markdown_string`), since those names are no longer module-level attributes — a function-local `from ... import` resolves the name fresh per call, so source-module patching is the only correct option.

Out of scope (explicit, separate maintainer decision): the optional `skill batch` mode. No schema change.
