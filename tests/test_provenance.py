"""Per-unit ingest provenance: the Writer stamps each unit with its run.

Covers the run lifecycle (``begin_run`` / ``finish_run`` / ``latest_config``)
and that ``persist_*`` stamps documents, summaries, and chunks with the open
run id — leaving them NULL when no run was opened (the skill-write path). End-
to-end stamping through ``scribe.main`` and config-drift warnings live in
``test_scribe.py`` and ``test_config.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import bartleby.config
import bartleby.db.connection
import bartleby.project
from bartleby import __version__
from bartleby.db.chunks import ChunkInput
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM, SCHEMA_VERSION
from bartleby.ingest.summarize import SummaryResult
from bartleby.ingest.writer import (
    ImageCaption,
    ParsedDocument,
    ParsedImage,
    Writer,
)


@pytest.fixture
def project_conn(tmp_path, monkeypatch):
    """A fresh project DB with PROJECTS_DIR / config isolated to tmp."""
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(bartleby.config, "BARTLEBY_DIR", tmp_path)
    monkeypatch.setattr(bartleby.config, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.config, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(bartleby.project, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.db.connection, "PROJECTS_DIR", projects)
    bartleby.project.create_project("prov")
    conn = open_db("prov")
    yield conn
    conn.close()


def _chunk(text: str, index: int = 0) -> ChunkInput:
    return ChunkInput(
        text=text, embedding=[0.0] * EMBEDDING_DIM, chunk_index=index,
    )


def _parsed(file_hash: str = "h1", chunks=None) -> ParsedDocument:
    return ParsedDocument(
        file_hash=file_hash,
        file_name="doc.txt",
        archive_path=Path("/archive/doc.txt"),
        page_count=None,
        token_count=10,
        document_chunks=chunks if chunks is not None else [_chunk("body text")],
        images=[],
    )


def _persist_image_doc(writer, conn, file_hash: str, image_hash: str) -> int:
    """Persist a one-image document and return the image's id."""
    writer.persist_parse(ParsedDocument(
        file_hash=file_hash, file_name="img.pdf",
        archive_path=Path("/a/img.pdf"),
        page_count=1, token_count=5, document_chunks=[],
        images=[ParsedImage(
            hash=image_hash, archive_path=Path("/a/img.jpg"),
            width=100, height=80, page_number=1, image_index_on_page=0,
        )],
    ))
    return conn.cursor().execute(
        "SELECT image_id FROM images WHERE file_hash = ?", (image_hash,)
    ).fetchone()[0]


def test_begin_run_records_redacted_snapshot(project_conn):
    writer = Writer(project_conn)
    run_id = writer.begin_run({"provider": "anthropic", "model": "x"})

    assert writer.run_id == run_id
    row = project_conn.cursor().execute(
        "SELECT config_json, bartleby_version, schema_version, finished_at "
        "FROM ingests WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    config_json, version, schema_version, finished_at = row
    # Sorted-key serialization, so a later drift check is a plain compare.
    assert config_json == '{"model": "x", "provider": "anthropic"}'
    assert version == __version__
    assert schema_version == SCHEMA_VERSION
    assert finished_at is None


def test_finish_run_sets_finished_at(project_conn):
    writer = Writer(project_conn)
    run_id = writer.begin_run({})
    writer.finish_run()
    finished_at = project_conn.cursor().execute(
        "SELECT finished_at FROM ingests WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    assert finished_at is not None


def test_finish_run_without_open_run_is_noop(project_conn):
    Writer(project_conn).finish_run()  # must not raise
    n = project_conn.cursor().execute("SELECT COUNT(*) FROM ingests").fetchone()[0]
    assert n == 0


def test_persist_parse_stamps_document_and_chunks(project_conn):
    writer = Writer(project_conn)
    run_id = writer.begin_run({})
    document_id = writer.persist_parse(_parsed())

    cur = project_conn.cursor()
    assert cur.execute(
        "SELECT ingest_run_id FROM documents WHERE document_id = ?",
        (document_id,),
    ).fetchone()[0] == run_id
    chunk_runs = [
        r[0] for r in cur.execute(
            "SELECT ingest_run_id FROM chunks WHERE source_kind = 'document'"
        )
    ]
    assert chunk_runs == [run_id]


def test_persist_summary_and_caption_stamp_run(project_conn):
    writer = Writer(project_conn)
    run_id = writer.begin_run({})
    document_id = writer.persist_parse(_parsed(chunks=[_chunk("body")]))
    # A second, image-bearing document so we have something to caption.
    image_id = _persist_image_doc(writer, project_conn, "h2", "imghash")

    writer.persist_caption(ImageCaption(
        image_id=image_id, analysis_json="{}", analysis_model="vlm-x",
        chunks=[_chunk("a captioned figure")],
    ))
    writer.persist_summary(
        document_id,
        SummaryResult(title="T", description="D", text="S", model="m",
                      truncated_from_tokens=None, authored_date=None),
        [_chunk("summary chunk")],
    )

    cur = project_conn.cursor()
    assert cur.execute(
        "SELECT ingest_run_id FROM summaries WHERE document_id = ?",
        (document_id,),
    ).fetchone()[0] == run_id
    # Both caption ('image') and summary chunks carry the run.
    other_runs = {
        r[0] for r in cur.execute(
            "SELECT DISTINCT ingest_run_id FROM chunks "
            "WHERE source_kind IN ('image', 'summary')"
        )
    }
    assert other_runs == {run_id}


def test_persist_caption_replay_is_noop(project_conn):
    # Replaying a caption (e.g. two concurrent scribe runs) must be a true
    # no-op: no second chunk insert, no UNIQUE-constraint crash, and the
    # original analysis left untouched (#308).
    writer = Writer(project_conn)
    writer.begin_run({})
    image_id = _persist_image_doc(writer, project_conn, "h3", "replayhash")

    writer.persist_caption(ImageCaption(
        image_id=image_id, analysis_json='{"first": true}',
        analysis_model="vlm-x", chunks=[_chunk("a captioned figure")],
    ))
    writer.persist_caption(ImageCaption(
        image_id=image_id, analysis_json='{"second": true}',
        analysis_model="vlm-y", chunks=[_chunk("a replayed caption")],
    ))  # must not raise

    cur = project_conn.cursor()
    analysis_json, analysis_model = cur.execute(
        "SELECT analysis_json, analysis_model FROM images WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    assert analysis_json == '{"first": true}'
    assert analysis_model == "vlm-x"
    texts = [r[0] for r in cur.execute(
        "SELECT text FROM chunks WHERE source_kind = 'image' AND source_id = ?",
        (image_id,),
    )]
    assert texts == ["a captioned figure"]


def test_no_run_leaves_provenance_null(project_conn):
    # The skill write path constructs a Writer but never opens a run.
    writer = Writer(project_conn)
    document_id = writer.persist_parse(_parsed())
    cur = project_conn.cursor()
    assert cur.execute(
        "SELECT ingest_run_id FROM documents WHERE document_id = ?",
        (document_id,),
    ).fetchone()[0] is None
    assert cur.execute(
        "SELECT ingest_run_id FROM chunks WHERE source_kind = 'document'"
    ).fetchone()[0] is None


def test_latest_config_returns_most_recent_prior(project_conn):
    writer = Writer(project_conn)
    assert writer.latest_config() is None  # no prior run yet

    writer.begin_run({"provider": "anthropic"})
    writer.finish_run()
    # A second Writer over the same DB sees the prior run's snapshot.
    assert Writer(project_conn).latest_config() == {"provider": "anthropic"}
