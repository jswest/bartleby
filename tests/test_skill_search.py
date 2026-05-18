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
    assert out["source_kinds"] == ["document"]
    assert out["memory_excluded"] is False
    # "pm25" appears only in alpha doc, chunk 0.
    texts = [r["text"] for r in out["results"]]
    assert any("pm25" in t for t in texts)


def test_search_returns_context_before_and_after(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "equity",
        "--context", "1",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["context"] == 1
    hit = next(r for r in out["results"] if "equity" in r["text"])
    # The "equity" chunk is chunk_index=1 in alpha; neighbors are 0 and 2.
    assert hit["context_before"] == ["alpha chunk zero about pm25"]
    assert hit["context_after"] == ["alpha chunk two on results"]


def test_search_context_zero_means_empty_arrays(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text", "equity",
        "--context", "0",
    ])
    out = json.loads(capsys.readouterr().out)
    for r in out["results"]:
        assert r["context_before"] == []
        assert r["context_after"] == []


def test_search_context_clamps_at_source_boundary(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text", "hello",
        "--context", "2",
    ])
    out = json.loads(capsys.readouterr().out)
    # "hello" is in beta chunk 0 (the start) — only chunk 1 follows.
    hit = next(r for r in out["results"] if r["source_name"] == "beta.txt")
    assert hit["context_before"] == []
    assert hit["context_after"] == ["beta chunk one says farewell"]


def test_search_context_does_not_cross_source_boundary(seeded_project, capsys):
    _run([
        "--project", seeded_project["project"],
        "--full-text",
        "concludes",
        "--context", "3",
    ])
    out = json.loads(capsys.readouterr().out)
    hit = next(r for r in out["results"] if "concludes" in r["text"])
    # alpha chunk 3 is the last in alpha — no chunks after it within alpha.
    assert hit["context_after"] == []
    # context_before pulls from earlier chunks in the same document only.
    assert all("alpha" in t for t in hit["context_before"])


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
            "INSERT INTO findings (session_id, title, body) VALUES (?, ?, ?)",
            (other_sess, "test", "body about pm25"),
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
            "--context", "99",
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
