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


class FakeS3Client:
    """In-memory stand-in for a boto3 S3 client: records every put_object."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.objects[(Bucket, Key)] = Body
        return {}


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
