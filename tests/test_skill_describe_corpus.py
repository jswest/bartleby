"""Smoke test for skill/scripts/describe_corpus.py."""

from __future__ import annotations

import json

from bartleby.db.connection import open_db
from bartleby.skill_scripts import describe_corpus
from tests._skill_fixtures import (  # noqa: F401
    project_env,
    seed_image,
    seeded_project,
)


def _run(project, capsys, extra=()):
    describe_corpus.main(["--project", project, *extra])
    return json.loads(capsys.readouterr().out)


def test_describe_corpus_empty(project_env, capsys):
    out = _run(project_env, capsys)
    assert out["document_count"] == 0
    assert out["chunk_count"] == 0
    assert out["token_count"] == 0
    assert out["authored_date"] == {
        "min": None, "max": None,
        "dated_document_count": 0, "undated_document_count": 0,
    }
    assert out["documents_by_year"] == []
    assert out["tags"] == []
    assert out["summary_coverage"] == {"summarized": 0, "unsummarized": 0}
    assert out["content_mix"] == []
    assert out["largest_documents"] == []


def test_describe_corpus_happy_path(seeded_project, capsys):
    # seeded_project: alpha.pdf (1000 tok, 4 'text' chunks, summary, no date)
    #                 beta.txt  (200 tok, 2 NULL-content_type chunks, no summary)
    out = _run(seeded_project["project"], capsys)

    assert out["document_count"] == 2
    assert out["chunk_count"] == 6
    assert out["token_count"] == 1200

    assert out["authored_date"] == {
        "min": None, "max": None,
        "dated_document_count": 0, "undated_document_count": 2,
    }
    assert out["documents_by_year"] == []
    assert out["tags"] == []
    assert out["summary_coverage"] == {"summarized": 1, "unsummarized": 1}

    mix = {row["content_type"]: row["chunk_count"] for row in out["content_mix"]}
    assert mix == {"text": 4, None: 2}

    # Largest first, by token_count.
    assert [d["id"] for d in out["largest_documents"]] == [
        seeded_project["doc_a"], seeded_project["doc_b"],
    ]
    assert out["largest_documents"][0] == {
        "id": seeded_project["doc_a"], "file_name": "alpha.pdf",
        "title": "Alpha", "token_count": 1000,
    }
    assert out["largest_documents"][1]["title"] is None  # beta has no summary


def test_describe_corpus_enriched(seeded_project, capsys):
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        # Give alpha's summary an authored_date.
        cur.execute(
            "UPDATE summaries SET authored_date = '2024-03-01' WHERE document_id = ?",
            (seeded_project["doc_a"],),
        )
        # A tag carried by one document.
        cur.execute("INSERT INTO tags (name, description) VALUES ('ch', 'Central Hudson')")
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
        # An image on beta adds two image chunks (image_ocr + image_description).
        seed_image(conn, seeded_project["doc_b"], file_hash="img1",
                   file_path="/tmp/img1.png")
    finally:
        conn.close()

    out = _run(project, capsys)

    assert out["authored_date"] == {
        "min": "2024-03-01", "max": "2024-03-01",
        "dated_document_count": 1, "undated_document_count": 1,
    }
    assert out["documents_by_year"] == [{"year": "2024", "document_count": 1}]
    assert out["tags"] == [{"name": "ch", "document_count": 1}]

    mix = {row["content_type"]: row["chunk_count"] for row in out["content_mix"]}
    assert mix == {"text": 4, None: 2, "image_ocr": 1, "image_description": 1}
    assert out["chunk_count"] == 8  # 6 + 2 image chunks


def test_describe_corpus_top_n(seeded_project, capsys):
    out = _run(seeded_project["project"], capsys, extra=["--top-n", "1"])
    assert len(out["largest_documents"]) == 1
    assert out["largest_documents"][0]["id"] == seeded_project["doc_a"]


def test_describe_corpus_rejects_nonpositive_top_n(seeded_project):
    # argparse rejects --top-n 0 before any work runs (guards SQLite's
    # LIMIT -1/0 footgun where a bogus value would dump the whole corpus).
    import pytest
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit):
            describe_corpus.main(["--project", seeded_project["project"],
                                  "--top-n", bad])
