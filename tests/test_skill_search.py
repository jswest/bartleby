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
    # Unfiltered search → no filters echo (uniform contract with scan/list_*).
    assert "filters" not in out
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


BRIEF_SEARCH_KEYS = {
    "chunk_id", "source_kind", "source_name", "page_number", "authored_date",
    "rank", "normalized_score", "text",
}


def test_search_brief_projection(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text", "pm25",
        "--brief",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["results"]
    for hit in out["results"]:
        assert set(hit) == BRIEF_SEARCH_KEYS
    assert out["modes"] == ["full-text"]  # envelope untouched


def _stamp_authored_date(seeded_project, document_id, value):
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "UPDATE summaries SET authored_date = ? WHERE document_id = ?",
            (value, document_id),
        )
    finally:
        conn.close()


def test_search_hit_carries_authored_date(seeded_project, capsys):
    """authored_date rides every hit (default + brief): the date when the doc is
    dated, null when undated. doc_a (alpha) gets a summary date; doc_b (beta) has
    no summary at all → undated."""
    _stamp_authored_date(seeded_project, seeded_project["doc_a"], "2024-05-09")
    # query → expected (term, source_name, authored_date)
    for term, name, expected in [
        ("pm25", "alpha.pdf", "2024-05-09"),  # dated doc_a
        ("hello", "beta.txt", None),          # undated doc_b
    ]:
        for extra in ([], ["--brief"]):
            _run([
                "--project", seeded_project["project"],
                "--full-text", term, *extra,
            ])
            out = json.loads(capsys.readouterr().out)
            hit = next(r for r in out["results"] if r["source_name"] == name)
            assert "authored_date" in hit  # present in every mode
            assert hit["authored_date"] == expected


def test_search_finding_authored_date_is_null(seeded_project, capsys):
    """Findings have no document anchor → authored_date is null, not missing."""
    from bartleby.session import start_session
    active = start_session(seeded_project["project"], memory_enabled=True)

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (active["session_id"], "test", "a one-line description",
             "body about pm25"),
        )
        finding_id = conn.last_insert_rowid()
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_finding_chunks(conn, finding_id, [
            ChunkInput(text="finding body about pm25", embedding=emb,
                       chunk_index=0),
        ])
    finally:
        conn.close()

    _run([
        "--project", seeded_project["project"],
        "--full-text", "--findings", "pm25",
    ])
    out = json.loads(capsys.readouterr().out)
    finding_hit = next(r for r in out["results"] if r["source_kind"] == "finding")
    assert finding_hit["authored_date"] is None


def test_search_brief_ignores_add_context(seeded_project, capsys):
    """--brief drops context even when --add-context is passed."""
    _run([
        "--project", seeded_project["project"],
        "--full-text", "equity",
        "--add-context", "1",
        "--brief",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["results"]
    for hit in out["results"]:
        assert set(hit) == BRIEF_SEARCH_KEYS


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
    active = start_session(seeded_project["project"], memory_enabled=False)

    # Seed a finding owned by the *active* no-memory session itself. search.py
    # drops the finding kind wholesale, so even the session's own findings are
    # excluded — a softened "exclude only foreign findings" impl must fail here.
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (active["session_id"], "test", "a one-line description", "body about pm25"),
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
    # No finding-kind result should appear — not even the active session's own.
    assert all(r["source_kind"] != "finding" for r in out["results"])


def test_search_findings_appear_under_memory_on(seeded_project, capsys):
    """Positive control: a finding-kind hit DOES surface for --findings when
    memory is on, so the exclusion assertions above can't pass vacuously."""
    from bartleby.session import start_session
    active = start_session(seeded_project["project"], memory_enabled=True)

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (active["session_id"], "test", "a one-line description", "body about pm25"),
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
    assert out["memory_excluded"] is False
    assert "finding" in out["source_kinds"]
    # The seeded finding is searchable and surfaces as a finding-kind result.
    assert any(r["source_kind"] == "finding" for r in out["results"])


def test_search_memory_excluded_false_without_findings_flag(seeded_project, capsys):
    """memory_excluded reports only *requested-and-dropped* findings: with no
    --findings, a memory-off session excludes nothing, so the flag stays False."""
    from bartleby.session import start_session
    start_session(seeded_project["project"], memory_enabled=False)

    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "pm25",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["memory_excluded"] is False
    assert "finding" not in out["source_kinds"]


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
        # The seeded summary already owns chunk_index 0 and 1, so append at 2
        # to avoid the (source_kind, source_id, chunk_index) UNIQUE collision.
        insert_summary_chunks(conn, summary_id, [
            ChunkInput(text="this is a summary-keyword chunk", embedding=emb, chunk_index=2),
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
    # Order: highest-score first, fully deterministic. 10 (1/61 + 1/62) just
    # edges out 30 (1/63 + 1/61), then 20/40 (a-only) and 50 (b-only) by their
    # single-list RRF contributions. Asserting the whole order — not just
    # "0th is 10 or 30" — catches a regression that reverses or reshuffles ties.
    ids_in_order = [cid for cid, _ in scored]
    assert ids_in_order == [10, 30, 20, 50, 40]


def test_fts_query_quotes_each_token():
    assert search_script._fts_query("pm25 equity") == '"pm25" "equity"'
    # Quote chars within a word are stripped
    assert search_script._fts_query('he"llo') == '"hello"'
    assert search_script._fts_query("   ") == ""


def test_fts_leg_skips_heading_only_term(seeded_project, capsys):
    """#464: the FTS leg is column-qualified to the body text, so a term that
    lives only in a chunk's section_heading ('Methods') returns no FTS hit,
    while a real body term ('equity') still does."""
    conn = open_db(seeded_project["project"])
    try:
        scope = {"document": None}
        # 'Methods' is the heading of doc_a chunk 1; it never appears in any body.
        assert search_script._fts_search(conn, "Methods", scope, 20) == []
        # Positive control: 'equity' is in that chunk's body text.
        hits = search_script._fts_search(conn, "equity", scope, 20)
        assert hits, "expected a body-text FTS hit for 'equity'"
    finally:
        conn.close()


def test_search_heading_only_term_no_fts_result(seeded_project, capsys):
    """End-to-end: a heading-only full-text query surfaces no body chunk whose
    text lacks the term (the keyword leg no longer injects heading hits)."""
    _run([
        "--project", seeded_project["project"],
        "--full-text", "Methods",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["modes"] == ["full-text"]
    # Full-text-only + a heading-only term ⇒ the keyword leg yields nothing.
    assert out["results"] == []


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
    assert out["filters"]["in_documents"] == [seeded_project["doc_b"]]
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
    assert out["filters"]["in_documents"] == [99999]
    assert out["results"] == []


def test_search_file_like_single_pattern(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--file-like", "alpha%",
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["alpha%"]
    assert out["results"]
    for r in out["results"]:
        assert r["source_id"] == seeded_project["doc_a"]


def test_search_file_like_excludes_nonmatching_doc(seeded_project, capsys):
    # "alpha" lives in doc_a (alpha.pdf), but we slice to beta.txt → no hits.
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--file-like", "beta%",
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["beta%"]
    assert out["results"] == []


def test_search_file_like_repeated_ors(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--file-like", "alpha%", "--file-like", "beta%",
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["alpha%", "beta%"]
    # The OR group admits both docs; "alpha" still only hits doc_a's chunks.
    assert out["results"]
    for r in out["results"]:
        assert r["source_id"] == seeded_project["doc_a"]


def test_search_file_like_ands_with_in_documents(seeded_project, capsys):
    # in-documents=doc_a AND file_like=beta% → empty intersection → no hits.
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "--in-documents", str(seeded_project["doc_a"]),
        "--file-like", "beta%",
        "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["in_documents"] == [seeded_project["doc_a"]]
    assert out["filters"]["file_like"] == ["beta%"]
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


# ---------- --returning projection (issue #419) ----------


def test_search_returning_projects_exact_fields(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"], "--full-text", "alpha",
        "--returning", "chunk_id,document_id,rank",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["results"]
    for r in out["results"]:
        assert list(r.keys()) == ["chunk_id", "document_id", "rank"]


def test_search_returning_document_id_is_honest_null_off_documents(seeded_project, capsys):
    """document_id is the source_id for a document-kind chunk; null elsewhere."""
    _run([
        "--project", seeded_project["project"], "--full-text", "alpha",
        "--summaries", "--documents",
        "--returning", "source_kind,source_id,document_id",
    ])
    out = json.loads(capsys.readouterr().out)
    by_kind = {}
    for r in out["results"]:
        by_kind.setdefault(r["source_kind"], []).append(r)
    for r in by_kind.get("document", []):
        assert r["document_id"] == r["source_id"]
    for r in by_kind.get("summary", []):
        assert r["document_id"] is None


def test_search_returning_overrides_brief(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"], "--full-text", "alpha",
        "--returning", "chunk_id", "--brief",
    ])
    out = json.loads(capsys.readouterr().out)
    for r in out["results"]:
        assert set(r) == {"chunk_id"}


def test_search_returning_unknown_field_errors(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        _run([
            "--project", seeded_project["project"], "--full-text", "alpha",
            "--returning", "chunk_id,nope",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"
    assert "document_id" in out["valid_fields"]


def test_search_returning_unknown_field_errors_on_zero_hits(seeded_project, capsys):
    """A typo'd --returning must error even when the query matches nothing,
    rather than coming back as a silent empty result set."""
    with pytest.raises(SystemExit) as exc:
        _run([
            "--project", seeded_project["project"], "--full-text",
            "zzzznomatchzzz", "--returning", "chunk_id,nope",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


def test_search_completes_when_source_deleted_underneath(seeded_project, capsys):
    """A chunk whose (source_kind, source_id) pair no longer resolves — its
    source row deleted by a concurrent session between fetch and name resolution
    — must degrade that one hit's source_name to "" rather than aborting the
    whole search with KeyError → INTERNAL_ERROR (issue #465)."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, "
            "page_count, token_count) VALUES (?, ?, ?, ?, ?)",
            ("h-ghost", "ghost.txt", "/tmp/ghost.txt", None, 0),
        )
        ghost_doc = conn.last_insert_rowid()
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_document_chunks(conn, ghost_doc, [
            ChunkInput(text="ghostword chunk", embedding=emb, chunk_index=0),
        ])
        # Drop the document row out from under its still-indexed chunk, so the
        # (document, ghost_doc) pair is absent from source_names' result.
        cur.execute("DELETE FROM documents WHERE document_id = ?", (ghost_doc,))
    finally:
        conn.close()

    # Default, brief, and --returning all read source_name — each must survive.
    for extra in ([], ["--brief"], ["--returning", "chunk_id,source_name"]):
        _run([
            "--project", seeded_project["project"],
            "--full-text", "ghostword", *extra,
        ])
        out = json.loads(capsys.readouterr().out)
        assert "error" not in out, f"search aborted with {extra}: {out}"
        hit = next(r for r in out["results"] if r["chunk_id"])
        assert hit["source_name"] == ""


def test_search_default_projection_unchanged_without_returning(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"], "--full-text", "alpha",
    ])
    out = json.loads(capsys.readouterr().out)
    hit = out["results"][0]
    # The full default hit shape is preserved byte-for-byte (no document_id leak).
    assert set(hit) == {
        "chunk_id", "source_kind", "source_id", "source_name", "file_name",
        "page_number", "authored_date", "chunk_index", "section_heading",
        "content_type", "text", "rank", "score", "normalized_score",
    }
