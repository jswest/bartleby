# `bartleby project publish <name> --to <s3-url>` ships a findings-free corpus copy to S3 (issue #519)

> Source: [#519](https://github.com/jswest/bartleby/issues/519)

`publish` exports a corpus as shareable raw material: a clean copy of the SQLite DB plus the original ingested files, content-addressed by `file_hash`, uploaded to an `s3://bucket/prefix` URL via `boto3`. The new `bartleby/share/` package holds two thin pieces — `s3.py` (put/get over a real boto3 client) and `publish.py` (copy + strip + gather + upload) — wired in through `bartleby/commands/project.py::publish` and the `project publish` parser in `cli.py`. No schema change.

## VACUUM INTO, never a live `cp`

Bartleby opens corpora in WAL mode (`_attach` sets `journal_mode = WAL`), so a raw byte copy of the `.db` file can be torn — it would miss un-checkpointed pages sitting in the `-wal` sidecar and capture a non-atomic mix of committed and in-flight state. `publish` instead opens the **source read-only** (`SQLITE_OPEN_READONLY`) and runs `VACUUM INTO ?`, which writes a fully-checkpointed, self-consistent snapshot from a single consistent read. This is also why the whole DB is carried as one file: we never cherry-pick the fts5/vec0 shadow tables — VACUUM INTO reproduces them faithfully, and the only per-row touches are the bounded sanitize below. After the strip (which runs in WAL on the copy) we `wal_checkpoint(TRUNCATE)` + `journal_mode = DELETE` so the uploaded `.db` is a single standalone file with no sidecars.

## The strip — on the copy only, source never mutated

A published corpus is permanently findings-free: everyone grows findings locally against the shared raw material, so the session layer is not optional to keep. On the copy, `strip_session_layer`:

1. nulls `document_tags.chunk_id` anchors pointing at finding chunks **before** deleting those chunks — `document_tags.chunk_id` references `chunks` with no cascade, so a live anchor would fail the delete on a FK violation. The document-level tag assignment (`document_id, tag_id, value`) survives; only the chunk anchor goes.
2. deletes the `source_kind='finding'` chunks and their `chunks_vec` rows via the typed `delete_chunks_of_kind` helper;
3. deletes `finding_citations`, `findings`, `sessions` (explicitly, though the FK cascade would chain from `sessions`, so intent reads plainly and a future FK change can't silently leave rows);
4. rebuilds the FTS index via `rebuild_fts`.

The source DB is opened read-only and is never written. Proven in `tests/test_share.py` by hashing the source file before and after `publish_project` and asserting byte-equality.

## Chunks discipline forced two helpers into `bartleby/db/chunks.py`

The polymorphic-chunks chokepoint (CLAUDE.md rule + the `test_chunks_chokepoint` static guard) forbids raw `INSERT INTO chunks_fts/chunks_vec/chunks` outside `bartleby/db/chunks.py`, and the guard exempts only the `integrity-check` command form. The FTS `'rebuild'` command and the bulk finding-chunk delete therefore belong in that sanctioned module, not in `share/publish.py`. We added two small maintenance helpers there — `delete_chunks_of_kind` (the whole-kind sibling of the existing `delete_chunks_for`) and `rebuild_fts` — and `publish.py` calls them. This is the one out-of-touches file the discipline required.

## Content-addressing by `file_hash`

Originals live in the project archive at `archive/<file_hash>/<file_hash><ext>` (documents) and `archive/images/<hash>.jpg` (images), with the archived path stored in `documents.file_path` / `images.file_path`. `gather_files` reads those rows off the stripped copy and maps `file_hash -> path`, uploading each as `files/<file_hash><ext>`. Anchor-split container rows whose archived file is absent on disk are skipped — their sections carry derived hashes pointing back at the same original, so the content-addressed set still covers every real artifact.

## Stubbed boto3 — no new test dependency

`s3._client()` returns a real `boto3.client("s3")` in production; `publish_project(..., client=...)` makes the client injectable. Tests pass an in-memory `FakeS3Client` recording every `put_object` and assert the `.db` + per-`file_hash` files round-trip — no `moto`, no pluggable local-dir backend, the transport layer stays thin. `boto3` is a new runtime dependency (added via `uv add boto3`) because `s3.py` imports it even though the client is stubbed under test. Suite green at 1067 passing.
