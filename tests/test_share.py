"""Tests for ``bartleby project publish`` (the bartleby.share package).

S3 is stubbed with an in-memory fake client — no moto, no new test dep. The
fixtures build a throwaway temp corpus (documents with on-disk originals, a
session + finding with a body chunk, and a tag anchored at that finding chunk)
so we can prove: the source DB is byte-identical before/after; the published
copy is findings-free (no findings/sessions/finding_citations rows, no
``source_kind='finding'`` chunks or their vec rows); a finding-anchored tag's
``chunk_id`` is nulled while the document-level assignment survives; and the
``.db`` plus the originals (keyed by ``file_hash``) all reach the stub.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import apsw
import pytest

import bartleby.project
from bartleby.db.chunks import ChunkInput, insert_document_chunks, insert_finding_chunks
from bartleby.db.connection import open_db, project_db_path
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.share import publish as publish_mod
from bartleby.share.s3 import parse_s3_url, S3Target


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + 0.001 * j for j in range(EMBEDDING_DIM)]


class _Body:
    """boto3's StreamingBody stand-in: a .read() that returns the stored bytes."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """In-memory stand-in for a boto3 S3 client.

    Records every ``put_object`` and serves them back via ``get_object`` so a
    publish -> import round-trip works against the same stub. No moto, no real
    boto3.
    """

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.objects[(Bucket, Key)] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError:
            raise KeyError(f"NoSuchKey: s3://{Bucket}/{Key}")
        return {"Body": _Body(data)}


@pytest.fixture
def published_corpus(tmp_path):
    """A temp corpus with originals, a finding, and a finding-anchored tag.

    Returns a dict with the project name, the archive paths, the finding chunk
    id, the document ids, and the surviving-tag bookkeeping the asserts need.
    """
    bartleby.project.create_project("pub")
    archive = bartleby.project.get_project_dir("pub") / "archive"

    # Two original files on disk, content-addressed by sha256.
    originals = {}
    for hsh_name, text in (("alpha", b"alpha original bytes"),
                           ("beta", b"beta original bytes")):
        file_hash = hashlib.sha256(text).hexdigest()
        dest_dir = archive / file_hash
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{file_hash}.pdf"
        dest.write_bytes(text)
        originals[hsh_name] = (file_hash, dest)

    conn = open_db("pub")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (originals["alpha"][0], "alpha.pdf", str(originals["alpha"][1]), 3, 500),
        )
        doc_a = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (originals["beta"][0], "beta.pdf", str(originals["beta"][1]), 1, 100),
        )
        doc_b = conn.last_insert_rowid()

        doc_chunk_ids = insert_document_chunks(conn, doc_a, [
            ChunkInput(text="alpha chunk zero", embedding=_emb(0.0), chunk_index=0),
            ChunkInput(text="alpha chunk one", embedding=_emb(0.1), chunk_index=1),
        ])

        # A session + finding with one body chunk, citing a document chunk.
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES ('s1', 1)"
        )
        session_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, 'F', 'hook', 'finding body')",
            (session_id,),
        )
        finding_id = conn.last_insert_rowid()
        finding_chunk_ids = insert_finding_chunks(conn, finding_id, [
            ChunkInput(text="finding body", embedding=_emb(9.0), chunk_index=0),
        ])
        finding_chunk_id = finding_chunk_ids[0]
        cur.execute(
            "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
            (finding_id, doc_chunk_ids[0]),
        )

        # A tag assigned to doc_a, anchored at the FINDING chunk. The
        # document-level assignment must survive; only the anchor is nulled.
        cur.execute(
            "INSERT INTO tags (name, description) VALUES ('topic', 'a topic tag')"
        )
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id, value, chunk_id) "
            "VALUES (?, ?, 'climate', ?)",
            (doc_a, tag_id, finding_chunk_id),
        )
    finally:
        conn.close()

    return {
        "project": "pub",
        "doc_a": doc_a,
        "doc_b": doc_b,
        "tag_id": tag_id,
        "finding_chunk_id": finding_chunk_id,
        "originals": originals,
    }


def _open_copy(db_bytes: bytes, tmp_path: Path) -> apsw.Connection:
    """Materialize an uploaded .db blob and open it with vec attached."""
    from bartleby.db.connection import _attach

    out = tmp_path / "downloaded.db"
    out.write_bytes(db_bytes)
    conn = apsw.Connection(str(out))
    _attach(conn)
    return conn


def test_source_db_byte_identical_after_publish(published_corpus):
    src = project_db_path("pub")
    before = src.read_bytes()
    before_digest = hashlib.sha256(before).hexdigest()

    publish_mod.publish_project("pub", "s3://bucket/corpora/pub",
                                client=FakeS3Client())

    after = src.read_bytes()
    assert hashlib.sha256(after).hexdigest() == before_digest
    assert after == before


def test_published_copy_is_findings_free(published_corpus, tmp_path):
    client = FakeS3Client()
    publish_mod.publish_project("pub", "s3://bucket/p", client=client)

    db_blob = client.objects[("bucket", "p/bartleby.db")]
    conn = _open_copy(db_blob, tmp_path)
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert cur.execute(
            "SELECT COUNT(*) FROM finding_citations"
        ).fetchone()[0] == 0
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind = 'finding'"
        ).fetchone()[0] == 0

        # The stripped finding chunk's vec row is gone too.
        fc = published_corpus["finding_chunk_id"]
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (fc,)
        ).fetchone()[0] == 0

        # Document chunks (and their vec rows) survive untouched.
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind = 'document'"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_finding_anchored_tag_nulled_assignment_survives(published_corpus, tmp_path):
    client = FakeS3Client()
    publish_mod.publish_project("pub", "s3://bucket/p", client=client)

    conn = _open_copy(client.objects[("bucket", "p/bartleby.db")], tmp_path)
    try:
        row = conn.cursor().execute(
            "SELECT value, chunk_id FROM document_tags "
            "WHERE document_id = ? AND tag_id = ?",
            (published_corpus["doc_a"], published_corpus["tag_id"]),
        ).fetchone()
        assert row is not None, "the document-level tag assignment must survive"
        value, chunk_id = row
        assert value == "climate"
        assert chunk_id is None, "the finding-anchored chunk_id must be nulled"
    finally:
        conn.close()


def test_files_uploaded_keyed_by_file_hash(published_corpus):
    client = FakeS3Client()
    result = publish_mod.publish_project("pub", "s3://bucket/p", client=client)

    assert result["file_count"] == 2
    for _name, (file_hash, dest) in published_corpus["originals"].items():
        key = ("bucket", f"p/files/{file_hash}.pdf")
        assert key in client.objects, f"missing upload for {file_hash}"
        assert client.objects[key] == dest.read_bytes()
        assert file_hash in result["file_urls"]


def test_stub_received_db_and_files(published_corpus):
    client = FakeS3Client()
    publish_mod.publish_project("pub", "s3://bucket/corpora/pub", client=client)

    keys = set(client.objects)
    assert ("bucket", "corpora/pub/bartleby.db") in keys
    # one .db + two files
    assert len(keys) == 3
    file_keys = {k for k in keys if k[1].startswith("corpora/pub/files/")}
    assert len(file_keys) == 2


def test_publish_missing_project_raises(project_env=None):
    bartleby.project.create_project("empty")
    # Remove the db to simulate a project with no corpus.
    project_db_path("empty").unlink()
    with pytest.raises(FileNotFoundError):
        publish_mod.publish_project("empty", "s3://bucket/p", client=FakeS3Client())


def test_parse_s3_url():
    t = parse_s3_url("s3://my-bucket/a/b/c")
    assert t == S3Target(bucket="my-bucket", prefix="a/b/c")
    assert t.key_for("x.db") == "a/b/c/x.db"
    assert parse_s3_url("s3://only-bucket").key_for("x.db") == "x.db"
    with pytest.raises(ValueError):
        parse_s3_url("https://example.com/x")
    with pytest.raises(ValueError):
        parse_s3_url("s3:///no-bucket")


# --------------------------------------------------------------------------- #
# import (issue #520) — round-trips against the same stubbed S3 client.
# --------------------------------------------------------------------------- #

from bartleby.lib.consts import EMBEDDING_MODEL  # noqa: E402
from bartleby.share import import_ as import_mod  # noqa: E402


def _publish_to_stub(project: str, url: str) -> FakeS3Client:
    """Publish ``project`` to ``url`` on a fresh stub, return the loaded stub."""
    client = FakeS3Client()
    publish_mod.publish_project(project, url, client=client)
    return client


def _rewrite_db_meta(client: FakeS3Client, bucket: str, key: str,
                     mutate, tmp_path: Path) -> None:
    """Materialize the stored ``.db`` blob, run ``mutate(conn)``, re-store it."""
    blob = client.objects[(bucket, key)]
    out = tmp_path / "mutate.db"
    out.write_bytes(blob)
    conn = apsw.Connection(str(out))
    try:
        with conn:
            mutate(conn.cursor())
    finally:
        conn.close()
    client.objects[(bucket, key)] = out.read_bytes()


def test_import_adopts_published_corpus_as_new_project(published_corpus, tmp_path):
    client = _publish_to_stub("pub", "s3://bucket/corpora/pub")

    result = import_mod.import_project("imported", "s3://bucket/corpora/pub",
                                       client=client)

    assert result["project"] == "imported"
    assert bartleby.project.get_active_project() == "imported"
    # The adopted DB opens under the strict gate (schema + vec attach) and
    # carries the document corpus as-is — no id rekey.
    conn = open_db("imported")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind = 'document'"
        ).fetchone()[0] == 2
        # Findings-free (publish stripped them); nothing to adopt there.
        assert cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0
    finally:
        conn.close()


def test_import_refuses_embedding_model_mismatch(published_corpus, tmp_path, capsys):
    client = _publish_to_stub("pub", "s3://bucket/corpora/pub")
    _rewrite_db_meta(
        client, "bucket", "corpora/pub/bartleby.db",
        lambda cur: cur.execute(
            "UPDATE meta SET value = 'some/other-model' WHERE key = 'embedding_model'"
        ),
        tmp_path,
    )

    with pytest.raises(import_mod.ImportRefused) as exc:
        import_mod.import_project("imported", "s3://bucket/corpora/pub",
                                  client=client)

    msg = str(exc.value)
    assert "some/other-model" in msg
    assert EMBEDDING_MODEL in msg
    # No side effects: the project was never registered.
    assert "imported" not in {p["name"] for p in bartleby.project.list_projects()}
    assert not import_mod.get_project_dir("imported").exists()


def test_import_refuses_missing_embedding_model_key(published_corpus, tmp_path):
    client = _publish_to_stub("pub", "s3://bucket/corpora/pub")
    _rewrite_db_meta(
        client, "bucket", "corpora/pub/bartleby.db",
        lambda cur: cur.execute("DELETE FROM meta WHERE key = 'embedding_model'"),
        tmp_path,
    )

    with pytest.raises(import_mod.ImportRefused):
        import_mod.import_project("imported", "s3://bucket/corpora/pub",
                                  client=client)
    assert not import_mod.get_project_dir("imported").exists()


def test_import_refuses_schema_version_mismatch(published_corpus, tmp_path):
    client = _publish_to_stub("pub", "s3://bucket/corpora/pub")
    _rewrite_db_meta(
        client, "bucket", "corpora/pub/bartleby.db",
        lambda cur: cur.execute(
            "UPDATE meta SET value = '1' WHERE key = 'schema_version'"
        ),
        tmp_path,
    )

    with pytest.raises(import_mod.ImportRefused) as exc:
        import_mod.import_project("imported", "s3://bucket/corpora/pub",
                                  client=client)
    assert "v1" in str(exc.value)
    assert not import_mod.get_project_dir("imported").exists()


def test_import_rewrites_file_paths_and_is_idempotent(published_corpus, tmp_path):
    client = _publish_to_stub("pub", "s3://bucket/corpora/pub")

    result = import_mod.import_project("imported", "s3://bucket/corpora/pub",
                                       client=client)
    assert result["file_count"] == 2

    archive = import_mod.get_project_dir("imported") / "archive"

    def landed_state():
        conn = open_db("imported")
        try:
            rows = conn.cursor().execute(
                "SELECT file_hash, file_path FROM documents ORDER BY file_hash"
            ).fetchall()
        finally:
            conn.close()
        return rows

    first = landed_state()
    for file_hash, file_path in first:
        p = Path(file_path)
        # Rewritten under the imported project's archive, addressed by file_hash.
        assert p.is_file()
        assert str(archive) in file_path
        assert file_hash in file_path
        # Bytes match what was published for that hash.
        published = client.objects[("bucket", f"corpora/pub/files/{file_hash}.pdf")]
        assert p.read_bytes() == published

    # Re-import: same hashes -> same landed paths -> identical state.
    import_mod.import_project("imported", "s3://bucket/corpora/pub",
                              client=client)
    assert landed_state() == first


def test_import_without_tags_drops_tags(published_corpus, tmp_path):
    client = _publish_to_stub("pub", "s3://bucket/corpora/pub")

    # Default: tags ride along.
    import_mod.import_project("with-tags", "s3://bucket/corpora/pub",
                              client=client)
    conn = open_db("with-tags")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags"
        ).fetchone()[0] == 1
    finally:
        conn.close()

    # --without-tags: tag definitions and assignments both gone.
    import_mod.import_project("no-tags", "s3://bucket/corpora/pub",
                              client=client, without_tags=True)
    conn = open_db("no-tags")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 0
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags"
        ).fetchone()[0] == 0
    finally:
        conn.close()
