# Critic pass 1 fixes for the publish→import contract (omnibus #494)

> Source: omnibus [#494](https://github.com/jswest/bartleby/issues/494) (v0.9.8),
> critic pass over the three merged sub-issues
> ([#517](https://github.com/jswest/bartleby/issues/517) embedding-model
> recording, [#519](https://github.com/jswest/bartleby/issues/519) `project
> publish`, [#520](https://github.com/jswest/bartleby/issues/520) `project
> import`).

A critic found four bugs in the merged publish→import seam, all in the same blast
radius: a corpus that lacks `meta.embedding_model` (every pre-#517 corpus) could
never become importable, and `import` could report success while leaving dangling
file references or crash with a boto3/apsw traceback. Additive only —
`SCHEMA_VERSION` stays at 9, no `--force` anywhere, no schema change.

## FIX 1 — the embedding-model backfill was unreachable for existing v9 corpora

#517 put the `embedding_model` backfill in the **always-runs tail** of
`bartleby.db.upgrades.upgrade()` (an idempotent `INSERT OR IGNORE`). But the CLI
wrapper `commands/project.py::upgrade` short-circuited *before* ever calling that
function: when `current == SCHEMA_VERSION` (9) it printed "already at schema v9.
Nothing to do." and `return`ed. So every corpus created **before** this omnibus
is at v9 with no `embedding_model` key and had **no CLI path to acquire it** — it
would `publish` a `.db` that `import` always hard-refuses, and the publisher (the
only one who can fix it) saw a green "Published".

The unit tests missed this because they call `upgrades.upgrade(conn, ...)`
directly, bypassing the CLI guard that contained the bug.

**Fix:** drop the early `return`. When `current == SCHEMA_VERSION` we now still
call `upgrades.upgrade(conn, current)` — the version-step `while` loop is a no-op
at that version, but the tail's `INSERT OR IGNORE embedding_model` +
`upgraded_at` write runs and backfills the key. The `current > SCHEMA_VERSION`
newer-than-code refusal path is preserved unchanged, and the console message
distinguishes "already at v9; ensured metadata is current" from a real version
bump. `upgrades.py` is untouched — its tail was already idempotent and correct;
only the CLI's reach to it was broken.

**Proof:** `tests/test_project.py::test_upgrade_backfills_embedding_model_on_current_v9`
goes through the **public command** (`project_cmd.upgrade(name=...)`, not
`upgrades.upgrade`) to exercise the actual seam that was dead: it deletes the
`embedding_model` key from a fresh v9 DB, runs the command, asserts the key is
back and equals `EMBEDDING_MODEL`, and that a second run neither errors nor
overwrites.

## FIX 2 — `import`'s `_land_files` swallowed real failures into a dangling success

`_land_files` had a bare `except Exception: continue`, and the `dest.write_bytes`
sat *outside* the try. So a transient S3 error (reset, throttle, expired
credentials), a `NoSuchKey`, **or** a disk-write failure mid-loop was silently
swallowed: the import exited 0 and reported success, but that row's
`documents`/`images` `file_path` still pointed at the **publisher's** absolute
path — dangling on the importer's machine. The web view (`web/.../queries.js`)
and `search.py` trust that column.

**Fix — narrow the catch, and abort loudly:**

- `write_bytes` (and the dir creation) is now **inside** the protected region, so
  a disk-write failure is caught rather than escaping mid-row.
- Only a genuinely-absent object — a boto3 `ClientError` whose code is
  `NoSuchKey`/`404`/`NoSuchBucket` (`_is_missing_key`) — is treated specially,
  and even then we do **not** silently succeed. Per the publish side, every
  gathered file actually exists in the artifact, so a missing object means the
  artifact is **corrupt**; we raise `ImportRefused` rather than leave a dangling
  row. (We chose refuse over collect-and-report: it's simpler and a published
  artifact missing a file it advertised is unambiguously broken.)
- **Any other exception** (the transient/credential/disk cases) re-raises, so the
  CLI maps it to a non-zero exit.

Because the project is *registered* (its directory created, `.db` moved into
place) before landing runs, a landing failure now tears the **freshly-created**
project back down before re-raising — mirroring the pre-verification scratch
cleanup. We track whether the project pre-existed and only `rmtree` one we
created, so a re-import that fails mid-landing never deletes a user's prior
project.

**Proof:**
`tests/test_share.py::test_import_aborts_on_non_missing_get_error_no_dangling_project`
wraps the in-memory `FakeS3Client` so the first file GET raises a non-`NoSuchKey`
`ClientError`; the import raises, no project is left registered, and the project
dir is gone (no row keeps a publisher path). A companion
`test_import_refuses_artifact_missing_advertised_file` covers the `NoSuchKey`
→ `ImportRefused` path. boto3 stays stubbed — no moto.

## FIX 3 — `publish` silently uploaded a dead artifact (composes with FIX 1)

A user who never ran `project upgrade` could `publish` a `.db` lacking
`embedding_model`; every importer hard-refuses it, but only the importer sees the
error. The publisher got a green "Published".

**Fix:** `publish.publish_project` now pre-flights the **source** corpus `meta`
for `embedding_model` *before* the expensive VACUUM/upload (via a read-only
`_read_source_meta` that never mutates the corpus). If the key is absent it
**refuses** with an actionable `ValueError` pointing at `bartleby project upgrade
<name>` — which, after FIX 1, actually backfills it. We deliberately do **not**
mutate the source to add the key; refuse-and-point keeps publish read-only on the
corpus and routes the user through the one place that owns the backfill.

**Proof:**
`tests/test_share.py::test_publish_refuses_source_without_embedding_model`
drops the key from a fresh corpus, asserts `publish_project` raises `ValueError`
mentioning `project upgrade`, and that **nothing** was uploaded to the stub
(`client.objects == {}`).

## FIX 4 — corrupt download / boto3 errors crashed instead of refusing cleanly

Two narrow-catch gaps:

- `import_._read_meta` caught only `apsw.SQLError`. A non-DB / garbage blob raises
  `apsw.NotADBError`, which is an `apsw.Error` but **not** a subclass of
  `SQLError`, so a corrupt download escaped as a traceback. Widened to
  `apsw.Error`, which routes a garbage blob through the same clean
  missing-`schema_version` refusal (`ImportRefused`).
- The command-layer handlers (`commands/project.py::publish` and `::import_`)
  caught only `ValueError`/`ImportRefused`, so a boto3 `NoCredentialsError` /
  `ClientError` / `EndpointConnectionError` (wrong URL, missing `bartleby.db` at
  the prefix, expired creds, unreachable endpoint) printed a traceback. Both
  handlers now catch those three botocore exceptions and emit a clean stderr
  error + non-zero exit. Kept minimal — we don't over-catch (programming errors
  still surface).

**Proof:** `tests/test_share.py::test_import_refuses_non_sqlite_blob` stores a
non-SQLite blob as `bartleby.db` and asserts a clean `ImportRefused` with no
project left behind. `test_publish_command_handler_maps_boto3_error_to_exit`
makes the handler's `publish_project` raise a `ClientError` and asserts a clean
`SystemExit(1)`.

## Deferred — surfaced to the maintainer, not fixed here

- **`import` overwriting a same-named existing project without confirmation.** The
  existing `test_import_rewrites_file_paths_and_is_idempotent` *encodes*
  overwrite-on-reimport as intended behavior; changing it is a design call. Left
  exactly as-is (and still passing).
- **Anchor-split corpora uploading N+1 copies of the same original file**
  (`gather_files` dedup defeat for #254 EDGAR sections). An efficiency issue that
  crosses the publish/ingest seam; tracked as a follow-up. `gather_files` is
  untouched by this pass.
