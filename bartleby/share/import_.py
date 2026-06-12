"""Import a published corpus from S3 as a fresh local project.

The mirror of :mod:`bartleby.share.publish`. ``import`` pulls the published
``.db`` plus the content-addressed originals down from an ``s3://bucket/prefix``
URL and adopts the ``.db`` **as-is** — no id rekeying, no merge into an existing
id space (that incremental-merge machinery is deliberately out of scope). The
imported corpus is raw material: publish already stripped the session layer, so
there is nothing findings-related to clean up here.

Two compatibility gates run **before** the corpus is registered or trusted, both
hard-refuse with no ``--force`` override:

1. **Schema version.** The source ``.db``'s ``meta.schema_version`` must equal
   this code's ``SCHEMA_VERSION`` (the same strict gate ``open_db`` enforces). A
   mismatch means the transported tables don't match this code's expectations.
2. **Embedding model.** The source ``.db``'s ``meta.embedding_model`` (the key
   #517 records) must equal the local pinned ``EMBEDDING_MODEL``. Vectors
   produced by a different model live in a different embedding space and are
   meaningless here, so a mismatch — *or a missing key, which we cannot verify*
   — is a hard refuse.

Both gates run against a temp copy of the ``.db`` downloaded into scratch, so a
refused import leaves **no** project directory and **no** side effects behind.

After both gates pass: the project directory is created, the verified ``.db`` is
moved into place, each original is downloaded by ``file_hash`` into the project
archive, and every ``documents``/``images`` ``file_path`` is rewritten to where
the file landed. Because the landing path is derived from the stable
``file_hash``, re-importing the same artifact is idempotent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import apsw
from botocore.exceptions import ClientError

from bartleby.db.connection import _attach, project_db_path
from bartleby.db.schema import SCHEMA_VERSION
from bartleby.lib.consts import EMBEDDING_MODEL
from bartleby.project import (
    get_project_dir,
    set_active_project,
    validate_project_name,
)
from bartleby.share import s3
from bartleby.share.publish import PUBLISHED_DB_NAME


class ImportRefused(Exception):
    """A compatibility gate failed; the import is refused with no side effects."""


def _read_meta(db_path: Path, key: str) -> str | None:
    """Read a single ``meta`` value from a raw (unregistered) ``.db``.

    Opens the file directly — no vec extension, no schema gate — because this
    runs *before* we trust the DB. Returns ``None`` if the key (or the ``meta``
    table itself) is absent, so the caller can refuse on a missing key.
    """
    conn = apsw.Connection(str(db_path))
    try:
        try:
            row = conn.cursor().execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        except apsw.Error:
            # A garbage / non-SQLite blob raises apsw.NotADBError, which is an
            # apsw.Error but NOT an apsw.SQLError — catching only the latter let
            # a corrupt download escape as a traceback instead of the clean
            # missing-key refusal path. Treat any DB-level error as "no value",
            # so _verify_compatible refuses it as a non-Bartleby corpus.
            return None
    finally:
        conn.close()
    return None if row is None else str(row[0])


def _verify_compatible(db_path: Path) -> None:
    """Run both compatibility gates against a downloaded ``.db``. Refuse on fail.

    Raises :class:`ImportRefused` (printing both relevant values) on a schema or
    embedding-model mismatch, or a missing ``embedding_model`` key. No
    ``--force`` path exists for either gate.
    """
    schema = _read_meta(db_path, "schema_version")
    if schema is None:
        raise ImportRefused(
            "Source database has no schema_version — not a Bartleby corpus, "
            "or a corrupt copy. Refusing to import."
        )
    if int(schema) != SCHEMA_VERSION:
        raise ImportRefused(
            f"Schema version mismatch: source database is v{schema}, this code "
            f"expects v{SCHEMA_VERSION}. Re-publish from a matching Bartleby "
            "version, or upgrade this install. Refusing to import."
        )

    model = _read_meta(db_path, "embedding_model")
    if model is None:
        raise ImportRefused(
            "Source database does not record its embedding model "
            "(meta.embedding_model is absent), so its vectors cannot be verified "
            f"against the local pinned model {EMBEDDING_MODEL!r}. Transported "
            "vectors in an unknown embedding space are meaningless. Refusing to "
            "import."
        )
    if model != EMBEDDING_MODEL:
        raise ImportRefused(
            f"Embedding model mismatch: source corpus was embedded with "
            f"{model!r} but this install is pinned to {EMBEDDING_MODEL!r}. "
            "Vectors live in a different embedding space and are meaningless "
            "here. Refusing to import."
        )


def _land_files(conn: apsw.Connection, target: s3.S3Target, archive: Path,
                client) -> int:
    """Download each original by ``file_hash`` and rewrite its ``file_path``.

    For every ``documents``/``images`` row, derives the published S3 key
    ``files/<file_hash><ext>`` from the recorded ``file_path``'s suffix, pulls
    the bytes, writes them into the project ``archive`` at a path derived from
    the stable ``file_hash``, and rewrites the row's ``file_path`` to that
    landed path. Idempotent: the landing path depends only on ``file_hash`` (and
    suffix), so a re-import overwrites in place to identical bytes.

    Error handling is deliberately strict: a row whose ``file_path`` is NOT
    rewritten still points at the *publisher's* absolute path, which is dangling
    on this machine — and the web view and search trust that column. So:

    * A genuinely-absent object (boto3 ``NoSuchKey``/404 ``ClientError``) is
      itself a sign of a corrupt artifact: publish only uploads files it
      actually found on disk, so a published artifact missing a file it
      advertised is broken. We refuse the whole import rather than silently
      leave a dangling row.
    * Any other failure (a transient S3 error, throttling, expired credentials,
      or a disk-write failure — all brought inside the protected region) aborts
      the import by re-raising, so the CLI maps it to a non-zero exit and the
      half-built project is cleaned up.

    Returns the count of files landed (every row, on success).
    """
    landed = 0
    cur = conn.cursor()
    for table, subdir in (("documents", None), ("images", "images")):
        rows = cur.execute(
            f"SELECT file_hash, file_path FROM {table}"
        ).fetchall()
        for file_hash, file_path in rows:
            ext = Path(file_path).suffix
            name = f"files/{file_hash}{ext}"
            try:
                data = s3.get_bytes(client, target, name)
                if subdir is None:
                    dest_dir = archive / file_hash
                else:
                    dest_dir = archive / subdir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{file_hash}{ext}"
                dest.write_bytes(data)
            except ClientError as e:
                if _is_missing_key(e):
                    raise ImportRefused(
                        f"Published artifact is missing a file it advertised "
                        f"(object {name!r} for {table} file_hash {file_hash}). "
                        "The artifact is corrupt — refusing to import a corpus "
                        "with dangling file references."
                    ) from e
                raise
            cur.execute(
                f"UPDATE {table} SET file_path = ? WHERE file_hash = ?",
                (str(dest), file_hash),
            )
            landed += 1
    return landed


def _is_missing_key(err: ClientError) -> bool:
    """True if a boto3 ``ClientError`` is an absent-object (NoSuchKey / 404)."""
    error = err.response.get("Error", {}) if hasattr(err, "response") else {}
    code = str(error.get("Code", ""))
    return code in ("NoSuchKey", "404", "NoSuchBucket")


def _drop_tags(conn: apsw.Connection) -> None:
    """Drop tag definitions and assignments from the adopted copy.

    Used by ``--without-tags``. The adopted ``.db`` carries tags by default
    (they ride along in the published copy); this removes both ``document_tags``
    assignments and the ``tags`` definitions. No anchor cleanup is needed —
    publish already nulled any finding-anchored ``document_tags.chunk_id``.
    """
    with conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_tags")
        cur.execute("DELETE FROM tags")


def import_project(name: str, from_url: str, *, client=None,
                   without_tags: bool = False, force: bool = False) -> dict:
    """Import the corpus published at ``from_url`` as local project ``name``.

    Downloads the published ``.db`` + originals from the ``s3://bucket/prefix``
    URL, verifies schema + embedding-model compatibility **before** registering
    anything, then adopts the ``.db`` as a fresh project, lands the originals by
    ``file_hash``, and rewrites their ``file_path``s. ``client`` is injectable so
    tests pass a stubbed boto3 client; production builds a real one.

    Raises :class:`ImportRefused` on a failed compatibility gate (no side
    effects), or when ``name`` already exists and ``force`` is not set (a
    same-name overwrite is opt-in because it drops the existing project's local
    findings); ``ValueError`` on a bad name/URL. With ``force``, re-importing an
    existing project overwrites it idempotently.
    """
    validate_project_name(name)
    target = s3.parse_s3_url(from_url)
    if client is None:
        client = s3._client()

    project_dir = get_project_dir(name)
    # Whether this project already existed before the import. A re-import
    # overwrites in place, so we must NOT delete a pre-existing project if
    # landing fails; a freshly-created one, however, is cleaned up so an aborted
    # import never leaves a half-registered project behind.
    preexisting = project_dir.exists()

    # A same-name collision is a hard stop unless ``force``: overwriting adopts a
    # new corpus over the existing project, dropping its local findings (which a
    # findings-free artifact cannot restore). Decide this up front, before any
    # download, so a refusal touches neither the network nor the existing project.
    if preexisting and not force:
        raise ImportRefused(
            f"Project '{name}' already exists; importing would overwrite it and "
            f"drop its local findings (a published artifact cannot restore them). "
            f"Pass --force to overwrite, or import under a different name."
        )

    # Download + verify in scratch first, so a refused import touches nothing.
    scratch = project_dir.parent / f".import-tmp-{name}"
    scratch.mkdir(parents=True, exist_ok=True)
    staged_db = scratch / PUBLISHED_DB_NAME
    try:
        staged_db.write_bytes(s3.get_bytes(client, target, PUBLISHED_DB_NAME))
        _verify_compatible(staged_db)

        # Gates passed — register the project. A re-import overwrites the prior
        # adopted DB; the archive is content-addressed so re-landing is a no-op
        # in content. Remove a stale archive so a dropped source file can't
        # linger.
        archive = project_dir / "archive"
        if archive.exists():
            shutil.rmtree(archive)
        project_dir.mkdir(parents=True, exist_ok=True)
        archive.mkdir(parents=True, exist_ok=True)

        dest_db = project_db_path(name)
        if dest_db.exists():
            dest_db.unlink()
        shutil.move(str(staged_db), str(dest_db))
    finally:
        for leftover in scratch.glob("*"):
            leftover.unlink()
        scratch.rmdir()

    # Landing can fail (a refused absent-key artifact, a transient S3 error, a
    # disk-write failure). The DB is registered now, so on ANY landing failure
    # tear the freshly-created project back down before re-raising — otherwise a
    # failed import leaves a project whose rows still point at the publisher's
    # dangling paths.
    try:
        conn = apsw.Connection(str(dest_db))
        try:
            _attach(conn)
            if without_tags:
                _drop_tags(conn)
            with conn:
                file_count = _land_files(conn, target, archive, client)
        finally:
            conn.close()
    except BaseException:
        if not preexisting and project_dir.exists():
            shutil.rmtree(project_dir)
        raise

    set_active_project(name)

    return {
        "project": name,
        "source": from_url,
        "file_count": file_count,
        "tags_dropped": without_tags,
    }
