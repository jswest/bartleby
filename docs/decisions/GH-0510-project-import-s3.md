# `bartleby project import <name> --from <s3-url>` adopts a published corpus as a fresh project (issue #520)

> Source: [#520](https://github.com/jswest/bartleby/issues/520) — part of omnibus
> [#494](https://github.com/jswest/bartleby/issues/494) (v0.9.8).

`import` is the mirror of `publish` (#519): it pulls the published `.db` plus the
content-addressed originals down from an `s3://bucket/prefix` URL and registers
them as a **new local project**. The new piece is `bartleby/share/import_.py`
(`import_` because `import` is a keyword), wired through
`bartleby/commands/project.py::import_` and the `project import` parser in
`cli.py`. `bartleby/share/s3.py` grows the get side (`get_bytes`, plus a thin
paginated `list_keys`) alongside the existing put side. No schema change —
`SCHEMA_VERSION` stays at 9.

## Adopt the whole `.db` as-is — no id rekey

The published `.db` is already a self-contained, findings-free corpus. `import`
adopts it **verbatim**: it moves the downloaded file into place as the project's
`bartleby.db` and never touches `chunks`, `documents.document_id`, or the
`(source_kind, source_id, chunk_index)` identity. Merging rows into an existing
id space (the `file_hash` + chunk-index rekey machinery) is explicitly deferred
and out of scope. Because we adopt rather than merge, we never raw-`INSERT` into
`chunks` — the polymorphic-chunks discipline is satisfied trivially by not
touching the table at all.

There is no session layer to clean up on import: publish already stripped
findings/sessions/finding_citations and nulled any finding-anchored
`document_tags.chunk_id`. The imported corpus is raw material everyone grows
findings against locally.

## Two compatibility gates, verified BEFORE registering — hard refuse, no `--force`

Both gates run against a temp copy of the `.db` downloaded into a scratch dir
(`.import-tmp-<name>` next to the projects dir), so a **refused import leaves no
project directory and no side effects**. Only after both pass does the project
dir get created and the `.db` moved into place.

1. **Schema version.** The source `.db`'s `meta.schema_version` must equal this
   code's `SCHEMA_VERSION` — the same strict gate `open_db` enforces. A mismatch
   means the transported tables don't match this code's expectations. We read the
   key off the raw (unregistered) file directly rather than going through
   `open_db`, because `open_db` requires the project to already be registered at
   `project_db_path(name)` — and we refuse *before* registering.
2. **Embedding model.** The source `.db`'s `meta.embedding_model` (the key #517
   records) must equal the local pinned `EMBEDDING_MODEL`
   (`bartleby/lib/consts.py`).

### Director call: hard refuse on embedding-model mismatch — and on a missing key

Vectors produced by a different embedding model live in a different embedding
space and are meaningless here, so a mismatch is a **hard refuse**: print both
model names, exit non-zero, do nothing else. There is deliberately **no `--force`
override** — a warn-and-proceed path would import vectors that silently retrieve
garbage. The same hard-refuse posture covers a source `.db` that is *missing* the
`embedding_model` key entirely: we cannot verify the embedding space, so we
refuse rather than trust it (a pre-#517 publish, or a non-Bartleby file). A
missing `schema_version` is likewise a refuse (not a Bartleby corpus, or a
corrupt copy).

## File-path rewrite by `file_hash` — idempotent re-import

`publish` uploads originals as `files/<file_hash><ext>`; the archived path in the
`.db` is the *publisher's* local path, meaningless on the importer's box. After
adopting, `import` walks `documents` and `images`, derives each object's key from
its `file_hash` + the recorded path's suffix, downloads the bytes, writes them
into the new project's `archive/` at a path derived **only from the stable
`file_hash`** (`archive/<hash>/<hash><ext>` for documents,
`archive/images/<hash><ext>` for images — mirroring the ingest archive layout),
and rewrites the row's `file_path` to that landed path.

Because the landing path is a pure function of `file_hash` (and suffix),
re-importing the same artifact overwrites in place to identical bytes and yields
identical `file_path` rows — **idempotent by `file_hash`**. A re-import also drops
the prior `archive/` first, so a file removed from the source can't linger. A row
whose published object is absent (an anchor-split container row publish skipped
because it holds no file of its own) is left with its path untouched rather than
failing the whole import — its sections carry derived hashes pointing back at the
same original.

## `--without-tags`

Tags ride along by default — they're in the adopted `.db`. `--without-tags`
deletes both `document_tags` assignments and the `tags` definitions on the
adopted copy. No anchor cleanup is needed: publish already nulled any
finding-anchored `document_tags.chunk_id`, so a surviving anchor points only at a
surviving chunk.

## Stubbed boto3 — no new test dependency

`tests/test_share.py`'s in-memory `FakeS3Client` (from #519) grew `get_object`
and `list_objects_v2` so a publish -> import round-trip works against the same
stub — no `moto`, no pluggable local-dir backend, the transport stays thin. (A
follow-up issue #521 will add `--from <local-path>`; it is not built here.) The
import tests prove: a well-formed artifact imports as a new project; an
embedding-model mismatch refuses with both names printed and no project
registered; a missing `embedding_model` key refuses; a `schema_version` mismatch
refuses; `file_path`s land under the new archive keyed by `file_hash` and a
second import is byte-for-byte idempotent; and `--without-tags` drops tags while
the default carries them. Suite green at 1077 passing.
