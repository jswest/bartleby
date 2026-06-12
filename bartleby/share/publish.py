"""Publish a findings-free copy of a corpus to S3.

The flow, source-never-mutated throughout:

1. ``VACUUM INTO`` a fresh ``.db`` copy of the corpus (never a live ``cp`` — the
   corpus runs WAL mode, so a raw byte copy can be torn). VACUUM INTO writes a
   clean, fully-checkpointed snapshot from a consistent read of the source.
2. Strip the session layer on the **copy**: delete ``findings``, ``sessions``,
   ``finding_citations``; delete the ``source_kind='finding'`` chunks and their
   ``chunks_vec`` rows; rebuild the FTS index; null any ``document_tags.chunk_id``
   anchor that pointed at a now-stripped chunk (the document-level tag assignment
   survives).
3. Gather the original ingested files, content-addressed by ``file_hash``.
4. Upload the ``.db`` and the files to the S3 URL.

A published corpus is raw material — everyone grows findings locally — so the
findings-free strip is the permanent rule, not an option.
"""

from __future__ import annotations

from pathlib import Path

import apsw

from bartleby.db.chunks import delete_chunks_of_kind, rebuild_fts
from bartleby.db.connection import _attach, project_db_path
from bartleby.project import get_project_dir, validate_project_name
from bartleby.share import s3


PUBLISHED_DB_NAME = "bartleby.db"


def _vacuum_into(source_db: Path, dest_db: Path) -> None:
    """Write a clean, consistent ``.db`` snapshot of ``source_db`` to ``dest_db``.

    Opens the source read-only and runs ``VACUUM INTO``. This is the whole-file
    transport: the entire DB (including the fts5/vec0 shadow tables) is carried,
    never cherry-picked. The source is only read, never written.
    """
    if dest_db.exists():
        dest_db.unlink()
    conn = apsw.Connection(str(source_db), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        conn.cursor().execute("VACUUM INTO ?", (str(dest_db),))
    finally:
        conn.close()


def strip_session_layer(conn: apsw.Connection) -> None:
    """Strip findings, sessions, and finding chunks from an opened copy.

    Operates on the publish copy only. Drops the finding chunks (and their
    ``chunks_vec`` rows) via the typed ``delete_chunks_of_kind`` helper, nulls
    ``document_tags.chunk_id`` anchors that pointed at them (the document-level
    assignment survives), clears the session layer, and rebuilds the FTS index
    over the surviving chunks through ``rebuild_fts``.
    """
    with conn:
        cur = conn.cursor()

        # Null any tag anchored at a finding chunk BEFORE the chunk is deleted:
        # ``document_tags.chunk_id`` references ``chunks`` with no cascade, so a
        # live anchor would block the delete on a FK violation. The document-level
        # assignment (document_id, tag_id, value) survives — only the anchor goes.
        cur.execute(
            "UPDATE document_tags SET chunk_id = NULL WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE source_kind = 'finding')"
        )

        delete_chunks_of_kind(conn, "finding")

        # FK chains (finding_citations -> findings -> sessions, all ON DELETE
        # CASCADE) mean deleting sessions is enough to clear findings and
        # citations, but we delete each explicitly so the intent reads plainly
        # and a future FK change can't silently leave rows behind.
        cur.execute("DELETE FROM finding_citations")
        cur.execute("DELETE FROM findings")
        cur.execute("DELETE FROM sessions")

        rebuild_fts(conn)


def gather_files(conn: apsw.Connection) -> dict[str, Path]:
    """Map ``file_hash`` -> on-disk path for every original ingested file.

    Reads from the (stripped) copy's ``documents`` and ``images`` tables. A
    container row from anchor-splitting holds no file of its own (its sections
    carry derived hashes pointing back at the same archived original), so rows
    whose archived path is missing on disk are skipped rather than failing the
    publish — the content-addressed set still covers every real artifact.
    """
    files: dict[str, Path] = {}
    cur = conn.cursor()
    for table in ("documents", "images"):
        for file_hash, file_path in cur.execute(
            f"SELECT file_hash, file_path FROM {table}"
        ):
            p = Path(file_path)
            if p.is_file():
                files[file_hash] = p
    return files


def publish_project(name: str, to_url: str, *, client=None) -> dict:
    """Publish project ``name`` to ``to_url`` (an ``s3://bucket/prefix`` URL).

    Returns a summary dict: the destination URL, the uploaded ``.db`` URL, and
    the per-``file_hash`` file URLs. ``client`` is injectable so tests pass a
    stubbed boto3 client; production builds a real one.

    The source corpus DB is opened read-only for the VACUUM INTO and is never
    mutated. All strip writes land on the copy.
    """
    validate_project_name(name)
    source_db = project_db_path(name)
    if not source_db.exists():
        raise FileNotFoundError(f"Project '{name}' has no database at {source_db}.")

    target = s3.parse_s3_url(to_url)
    if client is None:
        client = s3._client()

    # Build the copy inside the project's own scratch area, then clean it up.
    work_dir = get_project_dir(name) / ".publish-tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    copy_db = work_dir / PUBLISHED_DB_NAME

    try:
        _vacuum_into(source_db, copy_db)

        conn = apsw.Connection(str(copy_db))
        try:
            _attach(conn)
            strip_session_layer(conn)
            files = gather_files(conn)
        finally:
            conn.close()

        # Checkpoint + drop WAL so the uploaded .db is a single self-contained
        # file (the strip ran in WAL mode via _attach).
        wal = copy_db.with_name(copy_db.name + "-wal")
        shm = copy_db.with_name(copy_db.name + "-shm")
        if wal.exists() or shm.exists():
            ck = apsw.Connection(str(copy_db))
            try:
                ck.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
                ck.cursor().execute("PRAGMA journal_mode = DELETE")
            finally:
                ck.close()

        db_url = s3.put_file(client, target, PUBLISHED_DB_NAME, copy_db)

        file_urls: dict[str, str] = {}
        for file_hash, path in files.items():
            file_urls[file_hash] = s3.put_file(
                client, target, f"files/{file_hash}{path.suffix}", path
            )
    finally:
        for leftover in work_dir.glob("*"):
            leftover.unlink()
        work_dir.rmdir()

    return {
        "destination": to_url,
        "db_url": db_url,
        "file_count": len(file_urls),
        "file_urls": file_urls,
    }
