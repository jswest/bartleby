"""Smoke test for skill/scripts/describe_corpus.py."""

from __future__ import annotations

import json

from bartleby.db.connection import open_db
from bartleby.skill_scripts import describe_corpus
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seed_finding,
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
    assert out["chunk_length"] == {"median": None, "p90": None, "max": None}
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

    # Chunk-length shape over the 6 ingested chunks (lengths 26,26,27,27,28,28).
    assert out["chunk_length"] == {"median": 27, "p90": 28, "max": 28}

    # Largest first, by token_count.
    assert [d["id"] for d in out["largest_documents"]] == [
        f"document:{seeded_project['doc_a']}", f"document:{seeded_project['doc_b']}",
    ]
    assert out["largest_documents"][0] == {
        "id": f"document:{seeded_project['doc_a']}", "file_name": "alpha.pdf",
        "title": "Alpha", "token_count": 1000,
    }
    assert out["largest_documents"][1]["title"] is None  # beta has no summary


def test_describe_corpus_excludes_summary_and_finding_chunks(seeded_project, capsys):
    """chunk_count and content_mix count only document chunks. Adding more
    summary chunks and a finding chunk leaves both aggregates at the baseline —
    findings/summaries are polymorphic chunks but never corpus content."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        # Another summary, this one on beta, with its own chunk.
        from bartleby.db.chunks import ChunkInput, insert_summary_chunks
        from tests._skill_fixtures import _emb
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (seeded_project["doc_b"], "Beta", "desc", "summary body", "test"),
        )
        summary_b = conn.last_insert_rowid()
        insert_summary_chunks(conn, summary_b, [
            ChunkInput(text="beta summary chunk zero", embedding=_emb(4.0),
                       chunk_index=0),
        ])
        # A prior-session finding with a body chunk.
        cur.execute("INSERT INTO sessions (name) VALUES (?)", ("prior",))
        session_id = conn.last_insert_rowid()
        seed_finding(conn, session_id, title="Prior finding",
                     body="A finding note about the corpus.")
    finally:
        conn.close()

    out = _run(project, capsys)
    # Unchanged from test_describe_corpus_happy_path's baseline.
    assert out["chunk_count"] == 6
    mix = {row["content_type"]: row["chunk_count"] for row in out["content_mix"]}
    assert mix == {"text": 4, None: 2}


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
    assert out["largest_documents"][0]["id"] == f"document:{seeded_project['doc_a']}"


def test_describe_corpus_chunk_length_controlled(project_env, capsys):
    # Ten chunks of length 10,20,…,100 make the nearest-rank percentiles
    # unambiguous: median is the 5th value (50), p90 the 9th (90), max 100.
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from tests._skill_fixtures import _emb

    conn = open_db(project_env)
    try:
        conn.cursor().execute(
            "INSERT INTO documents (file_hash, file_name, file_path, token_count) "
            "VALUES ('h', 'doc.txt', '/tmp/doc.txt', 0)"
        )
        doc = conn.last_insert_rowid()
        insert_document_chunks(conn, doc, [
            ChunkInput(text="x" * (10 * (i + 1)), embedding=_emb(0.01 * i),
                       chunk_index=i)
            for i in range(10)
        ])
    finally:
        conn.close()

    out = _run(project_env, capsys)
    assert out["chunk_length"] == {"median": 50, "p90": 90, "max": 100}


def test_describe_corpus_chunk_length_scopes_to_slice(seeded_project, capsys):
    # Scoped to beta alone (chunk lengths 26, 28): stats narrow to that document.
    out = _run(seeded_project["project"], capsys,
               extra=["--in-documents", f"document:{seeded_project['doc_b']}"])
    assert out["chunk_length"] == {"median": 26, "p90": 28, "max": 28}


def test_describe_corpus_rejects_nonpositive_top_n(seeded_project):
    # argparse rejects --top-n 0 before any work runs (guards SQLite's
    # LIMIT -1/0 footgun where a bogus value would dump the whole corpus).
    import pytest
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit):
            describe_corpus.main(["--project", seeded_project["project"],
                                  "--top-n", bad])


# ---------- filtered describe_corpus (scope flags) ----------


def test_describe_corpus_unfiltered_omits_filters(seeded_project, capsys):
    out = _run(seeded_project["project"], capsys)
    assert "filters" not in out


def test_describe_corpus_scoped_to_one_document(seeded_project, capsys):
    # Scope to alpha alone: every aggregate narrows to that single document.
    out = _run(seeded_project["project"], capsys,
               extra=["--in-documents", f"document:{seeded_project['doc_a']}"])
    assert out["document_count"] == 1
    assert out["token_count"] == 1000
    assert out["chunk_count"] == 4
    assert {r["content_type"]: r["chunk_count"] for r in out["content_mix"]} == {"text": 4}
    assert out["summary_coverage"] == {"summarized": 1, "unsummarized": 0}
    assert [d["id"] for d in out["largest_documents"]] == [f"document:{seeded_project['doc_a']}"]
    assert out["authored_date"]["undated_document_count"] == 1  # alpha summary has no date
    assert out["filters"]["in_documents"] == [f"document:{seeded_project['doc_a']}"]
    assert out["filters"]["tags"] is None
    assert out["filters"]["excluded_null_dated"] == 0


def test_describe_corpus_tags_facet_lists_only_slice_tags(seeded_project, capsys):
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('ch', 'Central Hudson')")
        ch = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('nyseg', 'NYSEG')")
        nyseg = conn.last_insert_rowid()
        cur.execute("INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                    (seeded_project["doc_a"], ch))
        cur.execute("INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                    (seeded_project["doc_b"], nyseg))
    finally:
        conn.close()

    out = _run(project, capsys, extra=["--tag", "ch"])
    assert out["document_count"] == 1
    # nyseg lives only on beta, which is outside the ch slice → it's dropped.
    assert out["tags"] == [{"name": "ch", "document_count": 1}]
    assert out["filters"]["tags"] == ["ch"]


def test_describe_corpus_date_bound_reports_excluded_nulls(seeded_project, capsys):
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "UPDATE summaries SET authored_date = '2024-03-01' WHERE document_id = ?",
            (seeded_project["doc_a"],),
        )
    finally:
        conn.close()

    out = _run(project, capsys, extra=["--authored-after", "2024-01-01"])
    assert out["document_count"] == 1   # only dated alpha survives the bound
    assert out["authored_date"] == {
        "min": "2024-03-01", "max": "2024-03-01",
        "dated_document_count": 1, "undated_document_count": 0,
    }
    assert out["documents_by_year"] == [{"year": "2024", "document_count": 1}]
    # beta is undated → dropped by the bound, surfaced honestly.
    assert out["filters"]["excluded_null_dated"] == 1
    assert out["filters"]["authored_after"] == "2024-01-01"


def test_describe_corpus_date_bound_include_nulls_keeps_undated(seeded_project, capsys):
    """--include-nulls keeps undated documents under a date bound: beta rides
    along, both docs count, and nothing is reported as excluded."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "UPDATE summaries SET authored_date = '2024-03-01' WHERE document_id = ?",
            (seeded_project["doc_a"],),
        )
    finally:
        conn.close()

    out = _run(project, capsys,
               extra=["--authored-after", "2024-01-01", "--include-nulls"])
    assert out["document_count"] == 2   # dated alpha + undated beta both survive
    assert out["authored_date"] == {
        "min": "2024-03-01", "max": "2024-03-01",
        "dated_document_count": 1, "undated_document_count": 1,
    }
    assert out["filters"]["include_nulls"] is True
    assert out["filters"]["excluded_null_dated"] == 0


def test_describe_corpus_scoped_content_mix_reaches_images(seeded_project, capsys):
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        # An image on beta adds image_ocr + image_description chunks.
        seed_image(conn, seeded_project["doc_b"], file_hash="img1",
                   file_path="/tmp/img1.png")
    finally:
        conn.close()

    out = _run(project, capsys,
               extra=["--in-documents", f"document:{seeded_project['doc_b']}"])
    assert {r["content_type"]: r["chunk_count"] for r in out["content_mix"]} == {
        None: 2, "image_ocr": 1, "image_description": 1,
    }
    assert out["chunk_count"] == 4  # 2 text + 2 image chunks, all on beta


def test_describe_corpus_empty_slice_is_all_zeros_with_filters(seeded_project, capsys):
    out = _run(seeded_project["project"], capsys,
               extra=["--in-documents", "document:999999"])
    assert out["document_count"] == 0
    assert out["chunk_count"] == 0
    assert out["token_count"] == 0
    assert out["content_mix"] == []
    assert out["tags"] == []
    assert out["largest_documents"] == []
    assert out["authored_date"] == {
        "min": None, "max": None,
        "dated_document_count": 0, "undated_document_count": 0,
    }
    # Still self-describing — the filter that emptied the slice is echoed.
    assert out["filters"]["in_documents"] == ["document:999999"]


def test_describe_corpus_invalid_date_raises(seeded_project):
    import pytest
    with pytest.raises(SystemExit) as exc:
        describe_corpus.main(["--project", seeded_project["project"],
                              "--authored-after", "nonsense"])
    assert exc.value.code == 1


def test_describe_corpus_excludes_anchor_container_from_unsummarized(
    seeded_project, capsys
):
    """An anchor-split container (#254) owns zero chunks and owes no summary, so
    it must not inflate the unsummarized tally. seeded_project already has alpha
    (summarized) + beta (unsummarized); add a container with one section row.
    The section is an ordinary unsummarized document; the container is not."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM

    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        # Container: zero chunks, original file_hash, the four section columns NULL.
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, token_count) "
            "VALUES (?, ?, ?, ?)",
            ("filing", "filing.htm", "/tmp/filing.htm", 0),
        )
        container_id = conn.last_insert_rowid()
        # One section row pointing at the container, with its own chunk.
        cur.execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, token_count, "
            " parent_document_id, anchor_id, section_title, section_order) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("filing#sec1", "filing.htm", "/tmp/filing.htm", 50,
             container_id, "sec1", "Business", 0),
        )
        section_id = conn.last_insert_rowid()
        insert_document_chunks(conn, section_id, [
            ChunkInput(text="section body about the business operations here",
                       embedding=[0.2] * EMBEDDING_DIM, chunk_index=0),
        ])
    finally:
        conn.close()

    out = _run(project, capsys)
    # 4 docs: alpha (summarized), beta (unsummarized), container, section.
    assert out["document_count"] == 4
    # Container is excluded; only alpha is summarized; beta + section owe one.
    assert out["summary_coverage"] == {"summarized": 1, "unsummarized": 2}


def test_describe_corpus_summarized_container_not_double_subtracted(
    seeded_project, capsys
):
    """If a container ever acquires a summary row (an agent runs save_summary on
    it), it is already counted in ``summarized`` — it must not ALSO be subtracted
    as a container, or the unsummarized tally undercounts (possibly negative).
    The container is subtracted only when it has no summary row."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM

    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, token_count) "
            "VALUES (?, ?, ?, ?)",
            ("filing", "filing.htm", "/tmp/filing.htm", 0),
        )
        container_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, token_count, "
            " parent_document_id, anchor_id, section_title, section_order) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("filing#sec1", "filing.htm", "/tmp/filing.htm", 50,
             container_id, "sec1", "Business", 0),
        )
        section_id = conn.last_insert_rowid()
        insert_document_chunks(conn, section_id, [
            ChunkInput(text="section body about the business operations here",
                       embedding=[0.2] * EMBEDDING_DIM, chunk_index=0),
        ])
        # An agent summarized BOTH the section and the container.
        for doc_id, title in ((section_id, "Sec"), (container_id, "Filing")):
            cur.execute(
                "INSERT INTO summaries (document_id, title, description, text, model) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, "d", "t", "test"),
            )
    finally:
        conn.close()

    out = _run(project, capsys)
    # 4 docs; summarized = alpha + section + container = 3; only beta owes one.
    assert out["document_count"] == 4
    assert out["summary_coverage"] == {"summarized": 3, "unsummarized": 1}
