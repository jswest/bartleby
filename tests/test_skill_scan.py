"""Tests for skill/scripts/scan.py — the FTS5-only document filter."""

from __future__ import annotations

import json

import pytest

from bartleby.db.chunks import ChunkInput, insert_document_chunks, insert_summary_chunks
from bartleby.db.connection import open_db
from bartleby.skill_scripts import scan
from tests._skill_fixtures import _emb, project_env  # noqa: F401


MARKER = "I will divest my interests in"
# A marker chunk long enough to exceed the default 240-char preview.
LONG_MARKER = (
    MARKER + " ACME CORP within ninety days of confirmation. "
    + "This disclosure is provided pursuant to applicable ethics regulations. " * 3
)


def _run(corpus, args):
    scan.main(["--project", corpus["project"], *args])


@pytest.fixture
def scan_corpus(project_env):  # noqa: F811
    """Three templated filings sharing a marker phrase, plus distractors and a
    summary chunk that also contains the marker (to prove scan is docs-only)."""
    conn = open_db(project_env)
    try:
        cur = conn.cursor()

        def _doc(file_hash, file_name):
            cur.execute(
                "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (file_hash, file_name, f"/tmp/{file_name}", 2, 100),
            )
            return conn.last_insert_rowid()

        d1 = _doc("ha", "filing_a.txt")
        d2 = _doc("hb", "filing_b.txt")
        d3 = _doc("hc", "filing_c.txt")

        d1_ids = insert_document_chunks(conn, d1, [
            ChunkInput(text="SECTION 2 - ACME CORP", embedding=_emb(0.0),
                       chunk_index=0, section_heading="SECTION 2",
                       page_number=1, content_type="sec_text"),
            ChunkInput(text=LONG_MARKER, embedding=_emb(0.1),
                       chunk_index=1, page_number=1, content_type="sec_text"),
            ChunkInput(text="Certification: I have nothing further to report.",
                       embedding=_emb(0.2), chunk_index=2, page_number=2),
            ChunkInput(text=MARKER + " BETA LLC promptly.", embedding=_emb(0.3),
                       chunk_index=3, page_number=2),
        ])
        d2_ids = insert_document_chunks(conn, d2, [
            ChunkInput(text=MARKER + " GAMMA INC.", embedding=_emb(1.0),
                       chunk_index=0, page_number=1),
            ChunkInput(text="Some unrelated text about divest and interests scattered here.",
                       embedding=_emb(1.1), chunk_index=1, page_number=1),
        ])
        insert_document_chunks(conn, d3, [
            ChunkInput(text="Pure boilerplate certification with no marker.",
                       embedding=_emb(2.0), chunk_index=0, page_number=1),
        ])

        # A summary chunk that also contains the marker — scan must ignore it.
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (d1, "Filing A", "desc", "summary body", "test"),
        )
        summary_id = conn.last_insert_rowid()
        summary_ids = insert_summary_chunks(conn, summary_id, [
            ChunkInput(text="Summary: " + MARKER + " ACME CORP.", embedding=_emb(3.0),
                       chunk_index=0),
        ])
    finally:
        conn.close()

    return {
        "project": project_env,
        "d1": d1, "d2": d2, "d3": d3,
        "long_chunk_id": d1_ids[1],
        "distractor_chunk_id": d2_ids[1],
        "summary_chunk_id": summary_ids[0],
    }


def test_scan_phrase_enumerates_in_doc_order(scan_corpus, capsys):
    _run(scan_corpus, [MARKER])
    out = json.loads(capsys.readouterr().out)
    assert out["match_mode"] == "phrase"
    assert out["total"] == 3
    got = [(m["document_id"], m["chunk_index"]) for m in out["matches"]]
    assert got == [(scan_corpus["d1"], 1), (scan_corpus["d1"], 3), (scan_corpus["d2"], 0)]
    assert got == sorted(got)


def test_scan_pagination_stable_total(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--limit", "2", "--offset", "0"])
    p1 = json.loads(capsys.readouterr().out)
    assert p1["total"] == 3
    assert [(m["document_id"], m["chunk_index"]) for m in p1["matches"]] == [
        (scan_corpus["d1"], 1), (scan_corpus["d1"], 3),
    ]

    _run(scan_corpus, [MARKER, "--limit", "2", "--offset", "2"])
    p2 = json.loads(capsys.readouterr().out)
    assert p2["total"] == 3
    assert [(m["document_id"], m["chunk_index"]) for m in p2["matches"]] == [
        (scan_corpus["d2"], 0),
    ]


def test_scan_preview_truncates_by_default(scan_corpus, capsys):
    _run(scan_corpus, [MARKER])
    out = json.loads(capsys.readouterr().out)
    assert out["preview"] == 240
    long_match = next(m for m in out["matches"] if m["chunk_id"] == scan_corpus["long_chunk_id"])
    assert long_match["text_length"] > 240
    assert long_match["text"].endswith("…")
    assert len(long_match["text"]) == 241  # 240 chars + the ellipsis


def test_scan_preview_override_returns_full_text(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--preview", "600"])
    out = json.loads(capsys.readouterr().out)
    long_match = next(m for m in out["matches"] if m["chunk_id"] == scan_corpus["long_chunk_id"])
    assert not long_match["text"].endswith("…")
    assert len(long_match["text"]) == long_match["text_length"]


def test_scan_match_terms_is_superset_of_phrase(scan_corpus, capsys):
    _run(scan_corpus, ["divest my interests"])
    phrase_ids = {m["chunk_id"] for m in json.loads(capsys.readouterr().out)["matches"]}

    _run(scan_corpus, ["divest interests", "--match-terms"])
    terms_out = json.loads(capsys.readouterr().out)
    terms_ids = {m["chunk_id"] for m in terms_out["matches"]}

    assert terms_out["match_mode"] == "terms"
    assert phrase_ids < terms_ids  # strict superset
    # The non-contiguous distractor matches terms-AND but not the phrase.
    assert scan_corpus["distractor_chunk_id"] in terms_ids
    assert scan_corpus["distractor_chunk_id"] not in phrase_ids


def test_scan_in_documents_scope(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--in-documents", str(scan_corpus["d2"])])
    out = json.loads(capsys.readouterr().out)
    assert out["in_documents"] == [scan_corpus["d2"]]
    assert out["total"] == 1
    assert all(m["document_id"] == scan_corpus["d2"] for m in out["matches"])


def test_scan_excludes_summary_chunks(scan_corpus, capsys):
    _run(scan_corpus, [MARKER])
    out = json.loads(capsys.readouterr().out)
    ids = {m["chunk_id"] for m in out["matches"]}
    assert scan_corpus["summary_chunk_id"] not in ids
    assert out["total"] == 3


def test_scan_empty_query_errors(scan_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(scan_corpus, ["   "])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_QUERY"


def test_scan_tokenless_query_returns_no_matches(scan_corpus, capsys):
    _run(scan_corpus, ['"""'])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["matches"] == []


def test_scan_output_shape(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--in-documents", str(scan_corpus["d1"])])
    out = json.loads(capsys.readouterr().out)
    m = out["matches"][0]
    assert set(m.keys()) == {
        "document_id", "file_name", "chunk_id", "chunk_index",
        "page_number", "section_heading", "content_type", "text", "text_length",
    }
    assert m["file_name"] == "filing_a.txt"
    assert m["document_id"] == scan_corpus["d1"]
