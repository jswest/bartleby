"""Smoke tests for skill/scripts/search.py."""

from __future__ import annotations

import json
import struct

import pytest

from bartleby.skill_scripts import search as search_script
from bartleby.db.chunks import ChunkInput, insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch):
    """Replace the subprocess call to `bartleby embed` with an in-memory stub."""
    def _stub(query: str) -> bytes:
        # Just return a vector that's deterministic per-query; not actually
        # used to compute semantic order in our tests (we only verify modes
        # and shape).
        return struct.pack(f"{EMBEDDING_DIM}f", *[0.001] * EMBEDDING_DIM)
    monkeypatch.setattr(search_script, "_embed_query", _stub)


def _run(argv):
    search_script.main(argv)


def test_search_full_text_only_documents(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "pm25",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["modes"] == ["full-text"]
    assert out["source_kinds"] == ["document", "image"]
    assert out["memory_excluded"] is False
    # "pm25" appears only in alpha doc, chunk 0.
    texts = [r["text"] for r in out["results"]]
    assert any("pm25" in t for t in texts)


def test_search_returns_context_before_and_after(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "equity",
        "--add-context", "1",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["context"] == 1
    hit = next(r for r in out["results"] if "equity" in r["text"])
    # The "equity" chunk is chunk_index=1 in alpha; neighbors are 0 and 2.
    assert len(hit["context_before"]) == 1
    assert hit["context_before"][0]["text"] == "alpha chunk zero about pm25"
    assert hit["context_before"][0]["chunk_index"] == 0
    assert isinstance(hit["context_before"][0]["chunk_id"], int)
    assert len(hit["context_after"]) == 1
    assert hit["context_after"][0]["text"] == "alpha chunk two on results"
    assert hit["context_after"][0]["chunk_index"] == 2
    assert isinstance(hit["context_after"][0]["chunk_id"], int)


def test_search_context_entries_resolve_via_read_chunks(seeded_project, capsys):
    """A neighbor's chunk_id should round-trip through read_chunks."""
    from bartleby.skill_scripts import read_chunks
    _run([
        "--project", seeded_project["project"],
        "--full-text", "equity",
        "--add-context", "1",
    ])
    out = json.loads(capsys.readouterr().out)
    neighbor = next(
        r for r in out["results"] if "equity" in r["text"]
    )["context_after"][0]

    capsys.readouterr()  # drain previous output
    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", str(neighbor["chunk_id"]),
    ])
    fetched = json.loads(capsys.readouterr().out)
    assert fetched["chunks"][0]["text"] == neighbor["text"]


def test_search_default_context_omits_keys(seeded_project, capsys):
    """Default (no --add-context) omits the context keys entirely rather than
    shipping empty arrays."""
    _run([
        "--project", seeded_project["project"],
        "--full-text", "equity",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["context"] == 0
    assert out["results"]  # sanity: there are hits to check
    for r in out["results"]:
        assert "context_before" not in r
        assert "context_after" not in r


def test_search_context_clamps_at_source_boundary(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text", "hello",
        "--add-context", "2",
    ])
    out = json.loads(capsys.readouterr().out)
    # "hello" is in beta chunk 0 (the start) — only chunk 1 follows.
    hit = next(r for r in out["results"] if r["source_name"] == "beta.txt")
    assert hit["context_before"] == []
    assert len(hit["context_after"]) == 1
    assert hit["context_after"][0]["text"] == "beta chunk one says farewell"


def test_search_context_does_not_cross_source_boundary(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "concludes",
        "--add-context", "3",
    ])
    out = json.loads(capsys.readouterr().out)
    hit = next(r for r in out["results"] if "concludes" in r["text"])
    # alpha chunk 3 is the last in alpha — no chunks after it within alpha.
    assert hit["context_after"] == []
    # context_before pulls from earlier chunks in the same document only.
    assert all("alpha" in c["text"] for c in hit["context_before"])


def test_search_default_modes_are_semantic_and_fulltext(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "pm25",
    ])
    out = json.loads(capsys.readouterr().out)
    assert set(out["modes"]) == {"semantic", "full-text"}


def test_search_findings_excluded_under_no_memory(seeded_project, capsys):
    # Start a no-memory session and mark it active.
    from bartleby.session import start_session
    start_session(seeded_project["project"], memory_enabled=False)

    # Add a finding in a separate (memory-on) session so it can be excluded.
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("other-sess", 1),
        )
        other_sess = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (other_sess, "test", "a one-line description", "body about pm25"),
        )
        finding_id = conn.last_insert_rowid()
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_finding_chunks(conn, finding_id, [
            ChunkInput(text="finding body about pm25", embedding=emb, chunk_index=0),
        ])
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text", "--findings",
        "pm25",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["memory_excluded"] is True
    assert "finding" not in out["source_kinds"]
    # No finding-kind result should appear.
    assert all(r["source_kind"] != "finding" for r in out["results"])


def test_search_with_summaries_source_kind(seeded_project, capsys):
    # Seed a summary chunk that contains "summary-keyword" so FTS finds it.
    conn = open_db(seeded_project["project"])
    try:
        from bartleby.db.chunks import insert_summary_chunks
        cur = conn.cursor()
        summary_id = cur.execute(
            "SELECT summary_id FROM summaries WHERE document_id = ?",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
        emb = [0.1 * i for i in range(EMBEDDING_DIM)]
        insert_summary_chunks(conn, summary_id, [
            ChunkInput(text="this is a summary-keyword chunk", embedding=emb, chunk_index=0),
        ])
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text", "--summaries",
        "summary-keyword",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["source_kinds"] == ["summary"]
    assert any(r["source_kind"] == "summary" for r in out["results"])
    summary_hit = next(r for r in out["results"] if r["source_kind"] == "summary")
    assert summary_hit["source_name"] == "summary of alpha.pdf"


def test_search_context_out_of_range_rejected(seeded_project, capsys):
    with pytest.raises(SystemExit):
        _run([
            "--project", seeded_project["project"],
            "--full-text", "x",
            "--add-context", "99",
        ])


def test_search_empty_query_errors(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        _run([
            "--project", seeded_project["project"],
            "--full-text",
            "   ",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_QUERY"


def test_rrf_unit_known_lists():
    # Two ranked lists with known overlap; verify RRF math.
    a = [10, 20, 30, 40]   # ranks 1..4
    b = [30, 10, 50]       # ranks 1..3
    scored = search_script._rrf([a, b], k=60)
    by_id = dict(scored)
    # 30: 1/(60+3) [from a] + 1/(60+1) [from b]
    expected_30 = 1.0 / 63 + 1.0 / 61
    expected_10 = 1.0 / 61 + 1.0 / 62
    assert by_id[30] == pytest.approx(expected_30)
    assert by_id[10] == pytest.approx(expected_10)
    # Order: highest-score first.
    ids_in_order = [cid for cid, _ in scored]
    assert ids_in_order[0] in (10, 30)


def test_fts_query_quotes_each_token():
    assert search_script._fts_query("pm25 equity") == '"pm25" "equity"'
    # Quote chars within a word are stripped
    assert search_script._fts_query('he"llo') == '"hello"'
    assert search_script._fts_query("   ") == ""


def test_search_results_have_rank_and_normalized_score(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text", "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["results"], "expected at least one hit for 'alpha'"
    # rank is 1-indexed and consecutive.
    assert [r["rank"] for r in out["results"]] == list(
        range(1, len(out["results"]) + 1)
    )
    # First result has normalized_score == 1.0; raw score is positive.
    assert out["results"][0]["normalized_score"] == pytest.approx(1.0)
    assert out["results"][0]["score"] > 0
    # All normalized scores in (0, 1].
    for r in out["results"]:
        assert 0 < r["normalized_score"] <= 1.0


def test_search_in_documents_filters_to_listed_docs(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--in-documents", str(seeded_project["doc_b"]),
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    # "alpha" appears in doc_a's chunks, but we scoped to doc_b — no hits.
    assert out["in_documents"] == [seeded_project["doc_b"]]
    assert out["results"] == []


def test_search_in_documents_returns_hits_in_scope(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--in-documents", str(seeded_project["doc_a"]),
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["results"]
    for r in out["results"]:
        assert r["source_kind"] == "document"
        assert r["source_id"] == seeded_project["doc_a"]


def test_search_in_documents_drops_findings(seeded_project, capsys):
    # Seed a finding so we can verify it gets dropped even when --findings asked.
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("with-findings", 1),
        )
        sess = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (sess, "f", "f", "body talking about alpha"),
        )
        finding_id = conn.last_insert_rowid()
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_finding_chunks(conn, finding_id, [
            ChunkInput(text="finding body about alpha", embedding=emb, chunk_index=0),
        ])
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text", "--findings",
        "--in-documents", str(seeded_project["doc_a"]),
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert "finding" not in out["source_kinds"]
    assert all(r["source_kind"] != "finding" for r in out["results"])


def test_search_in_documents_invalid_ids_returns_empty(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--in-documents", "99999",
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["in_documents"] == [99999]
    assert out["results"] == []


def test_search_semantic_scope_survives_global_nearest_elsewhere(
    seeded_project, capsys, monkeypatch
):
    """Regression for #55: when more out-of-scope chunks than the over-fetch
    window are nearer to the query than the in-scope chunks, a semantic search
    scoped to doc_a must still return doc_a's chunks. The old post-filter
    (fetch k globally-nearest, then drop out-of-scope rows) starved to [] here;
    pushing the scope into the index fixes it."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks

    # Seed a decoy document with MORE near-query chunks than the over-fetch
    # window, so every chunk in the global top-`overfetch` is out of scope and a
    # post-filter to doc_a would wipe the result set to empty.
    limit = 1  # mirror work()'s overfetch calc for this query's --limit
    overfetch = max(
        limit * search_script.OVERFETCH_MULTIPLIER, search_script.OVERFETCH_FLOOR
    )
    n_decoys = overfetch + 5
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("h-decoy", "decoy.txt", "/tmp/decoy.txt", None, 0),
        )
        doc_c = conn.last_insert_rowid()
        # Decoys cluster at seed ~1.0; doc_a's fixture chunks sit at seed 0.0-0.3.
        insert_document_chunks(conn, doc_c, [
            ChunkInput(
                text=f"decoy chunk {i}",
                embedding=[1.0 + 0.0001 * i + 0.001 * j for j in range(EMBEDDING_DIM)],
                chunk_index=i,
            )
            for i in range(n_decoys)
        ])
    finally:
        conn.close()

    # Query vector near 1.05 ranks the decoys (and doc_b) as the global nearest;
    # doc_a's chunks fall well outside the top-`overfetch` window.
    near_decoys = struct.pack(
        f"{EMBEDDING_DIM}f", *[1.05 + 0.001 * j for j in range(EMBEDDING_DIM)]
    )
    monkeypatch.setattr(search_script, "_embed_query", lambda q: near_decoys)

    # Premise: unscoped, the global nearest hit is NOT in doc_a.
    _run([
        "--project", seeded_project["project"],
        "--semantic", "--documents", "--limit", str(limit),
        "anything",
    ])
    unscoped = json.loads(capsys.readouterr().out)
    assert unscoped["results"][0]["source_id"] != seeded_project["doc_a"]

    # Scoped to doc_a, the in-scope chunks must come back — not [].
    _run([
        "--project", seeded_project["project"],
        "--semantic", "--documents", "--limit", str(limit),
        "--in-documents", str(seeded_project["doc_a"]),
        "anything",
    ])
    scoped = json.loads(capsys.readouterr().out)
    assert scoped["results"], "scoped semantic search starved to empty (issue #55)"
    for r in scoped["results"]:
        assert r["source_kind"] == "document"
        assert r["source_id"] == seeded_project["doc_a"]


def test_search_includes_image_chunks_with_id_and_path(seeded_project, capsys):
    from tests._skill_fixtures import seed_image
    conn = open_db(seeded_project["project"])
    try:
        image_id = seed_image(
            conn, seeded_project["doc_a"],
            file_hash="img-hash-1", file_path="images/img-hash-1.jpg",
            description="A bar chart showing pm25 levels over time.",
            ocr_text="figure caption",
        )
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "pm25",
    ])
    out = json.loads(capsys.readouterr().out)
    image_hits = [r for r in out["results"] if r["source_kind"] == "image"]
    assert image_hits, "expected an image hit for 'pm25' in the description"
    hit = image_hits[0]
    assert hit["source_id"] == image_id
    assert hit["image_id"] == image_id
    assert hit["image_file_path"] == "images/img-hash-1.jpg"
    # source_name renders the document anchor.
    assert "image in alpha.pdf" in hit["source_name"]


def test_search_in_documents_filters_image_chunks(seeded_project, capsys):
    from tests._skill_fixtures import seed_image
    conn = open_db(seeded_project["project"])
    try:
        seed_image(
            conn, seeded_project["doc_a"],
            file_hash="img-in-a", file_path="images/a.jpg",
            description="alpha-only image",
        )
        seed_image(
            conn, seeded_project["doc_b"],
            file_hash="img-in-b", file_path="images/b.jpg",
            description="beta-only image",
        )
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text", "--images",
        "--in-documents", str(seeded_project["doc_a"]),
        "image",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["source_kinds"] == ["image"]
    assert out["results"], "expected at least one image hit in doc_a's scope"
    for r in out["results"]:
        assert r["source_kind"] == "image"
        assert "image in alpha.pdf" in r["source_name"]


def test_search_hits_include_file_name_and_page_number(seeded_project, capsys):
    """Hits should carry first-class file_name + page_number fields."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    conn = open_db(seeded_project["project"])
    try:
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        # Add a chunk with a first-class page_number.
        insert_document_chunks(conn, seeded_project["doc_b"], [
            ChunkInput(
                text="beta page-seven keyword",
                embedding=emb, chunk_index=2,
                section_heading=None, page_number=7, content_type="text",
            ),
        ])
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "page-seven",
    ])
    out = json.loads(capsys.readouterr().out)
    [hit] = out["results"]
    assert hit["file_name"] == "beta.txt"
    assert hit["page_number"] == 7


def test_search_image_source_name_notes_multiple_docs(seeded_project, capsys):
    """An image attached to two documents shows '+1 other docs' in source_name."""
    from tests._skill_fixtures import seed_image
    from bartleby.db.chunks import ChunkInput, insert_image_chunks
    conn = open_db(seeded_project["project"])
    try:
        image_id = seed_image(
            conn, seeded_project["doc_a"],
            file_hash="shared-img", file_path="images/shared.jpg",
            description="a shared image referenced everywhere",
        )
        # Attach the same image to doc_b too.
        conn.cursor().execute(
            "INSERT INTO document_images "
            "(document_id, image_id, page_number, image_index_on_page) "
            "VALUES (?, ?, ?, ?)",
            (seeded_project["doc_b"], image_id, 1, 1),
        )
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "shared",
    ])
    out = json.loads(capsys.readouterr().out)
    image_hits = [r for r in out["results"] if r["source_kind"] == "image"]
    assert image_hits
    name = image_hits[0]["source_name"]
    assert "image in alpha.pdf" in name
    assert "+1 other docs" in name
