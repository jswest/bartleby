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


def test_scan_brief_keeps_only_locators(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--brief"])
    out = json.loads(capsys.readouterr().out)
    assert out["matches"]
    assert out["total"] == 3  # envelope/total unchanged
    for m in out["matches"]:
        assert set(m) == {
            "document_id", "file_name", "chunk_id", "page_number",
            "authored_date",
        }


def test_scan_brief_ignores_preview(scan_corpus, capsys):
    """--brief drops text entirely, so --preview has nothing to truncate."""
    _run(scan_corpus, [MARKER, "--brief", "--preview", "10"])
    out = json.loads(capsys.readouterr().out)
    for m in out["matches"]:
        assert "text" not in m
        assert "text_length" not in m


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
    assert out["filters"]["in_documents"] == [scan_corpus["d2"]]
    assert out["total"] == 1
    assert all(m["document_id"] == scan_corpus["d2"] for m in out["matches"])


def test_scan_file_like_single_pattern(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--file-like", "filing_a%"])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["filing_a%"]
    assert out["total"] == 2  # the two marker chunks in filing_a.txt
    assert all(m["document_id"] == scan_corpus["d1"] for m in out["matches"])


def test_scan_file_like_repeated_ors(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--file-like", "filing_a%", "--file-like", "filing_b%"])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["filing_a%", "filing_b%"]
    # filing_a (2 markers) OR filing_b (1 marker); filing_c has none.
    assert out["total"] == 3
    assert {m["document_id"] for m in out["matches"]} == {
        scan_corpus["d1"], scan_corpus["d2"],
    }


def test_scan_file_like_ands_with_in_documents(scan_corpus, capsys):
    # in-documents={d1,d2} AND file_like=filing_b% → only d2 survives.
    _run(scan_corpus, [
        MARKER,
        "--in-documents", f"{scan_corpus['d1']},{scan_corpus['d2']}",
        "--file-like", "filing_b%",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["filing_b%"]
    assert out["total"] == 1
    assert all(m["document_id"] == scan_corpus["d2"] for m in out["matches"])


def test_scan_file_like_no_match_empties(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--file-like", "nonexistent%"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["matches"] == []
    assert out["filters"]["file_like"] == ["nonexistent%"]


# ---------- --heading-like (chunk-level section_heading filter) ----------


@pytest.fixture
def heading_corpus(project_env):  # noqa: F811
    """One doc whose marker chunks each sit under a distinct section_heading,
    plus a marker chunk with no heading — so --heading-like can prove it keeps
    only matching headings and never the NULL-heading chunk."""
    conn = open_db(project_env)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("hh", "report.txt", "/tmp/report.txt", 3, 100),
        )
        doc = conn.last_insert_rowid()
        ids = insert_document_chunks(conn, doc, [
            ChunkInput(text=MARKER + " under Q1.", embedding=_emb(0.0),
                       chunk_index=0, section_heading="2023 Q1", page_number=1),
            ChunkInput(text=MARKER + " under Q2.", embedding=_emb(0.1),
                       chunk_index=1, section_heading="2023 Q2", page_number=2),
            ChunkInput(text=MARKER + " with no heading.", embedding=_emb(0.2),
                       chunk_index=2, page_number=3),
        ])
    finally:
        conn.close()
    return {"project": project_env, "doc": doc,
            "q1_id": ids[0], "q2_id": ids[1], "no_heading_id": ids[2]}


def test_scan_default_match_carries_section_heading(heading_corpus, capsys):
    _run(heading_corpus, [MARKER])
    out = json.loads(capsys.readouterr().out)
    headings = {m["chunk_id"]: m["section_heading"] for m in out["matches"]}
    assert headings[heading_corpus["q1_id"]] == "2023 Q1"
    assert headings[heading_corpus["no_heading_id"]] is None


def test_scan_brief_drops_section_heading(heading_corpus, capsys):
    _run(heading_corpus, [MARKER, "--brief"])
    out = json.loads(capsys.readouterr().out)
    assert out["matches"]
    for m in out["matches"]:
        assert "section_heading" not in m


def test_scan_heading_like_single_pattern(heading_corpus, capsys):
    _run(heading_corpus, [MARKER, "--heading-like", "2023 Q1"])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["heading_like"] == ["2023 Q1"]
    assert out["total"] == 1
    assert [m["chunk_id"] for m in out["matches"]] == [heading_corpus["q1_id"]]
    # The NULL-heading marker chunk never rides along.
    assert heading_corpus["no_heading_id"] not in {m["chunk_id"] for m in out["matches"]}


def test_scan_heading_like_repeated_ors(heading_corpus, capsys):
    _run(heading_corpus, [MARKER, "--heading-like", "2023 Q1", "--heading-like", "2023 Q2"])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["heading_like"] == ["2023 Q1", "2023 Q2"]
    assert out["total"] == 2
    assert {m["chunk_id"] for m in out["matches"]} == {
        heading_corpus["q1_id"], heading_corpus["q2_id"],
    }


def test_scan_heading_like_wildcard_ands_with_file_like(heading_corpus, capsys):
    # file_like matches the doc; heading_like '2023 Q%' keeps only the two
    # quarter chunks (AND drops the NULL-heading one).
    _run(heading_corpus, [MARKER, "--file-like", "report%", "--heading-like", "2023 Q%"])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == ["report%"]
    assert out["filters"]["heading_like"] == ["2023 Q%"]
    assert out["total"] == 2
    assert {m["chunk_id"] for m in out["matches"]} == {
        heading_corpus["q1_id"], heading_corpus["q2_id"],
    }


def test_scan_heading_like_no_match_empties(heading_corpus, capsys):
    _run(heading_corpus, [MARKER, "--heading-like", "1999 Q%"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["matches"] == []
    assert out["filters"]["heading_like"] == ["1999 Q%"]


def test_scan_heading_like_echoed_when_no_other_scope(heading_corpus, capsys):
    """--heading-like alone is enough to surface a filters object (it isn't a
    Scope field, so the echo must create one)."""
    _run(heading_corpus, [MARKER, "--heading-like", "2023 Q1"])
    out = json.loads(capsys.readouterr().out)
    assert "filters" in out
    assert out["filters"] == {"heading_like": ["2023 Q1"]}


def test_scan_heading_like_count_by_document(heading_corpus, capsys):
    """--heading-like composes with --count-by document (chunk-level pushdown)."""
    _run(heading_corpus, [MARKER, "--count-by", "document", "--heading-like", "2023 Q%"])
    out = json.loads(capsys.readouterr().out)
    assert out["distinct_document_count"] == 1
    assert out["total_chunk_count"] == 2  # only the two headed chunks
    assert out["filters"]["heading_like"] == ["2023 Q%"]


# ---------- text-only MATCH: heading-only terms are not body hits (#464) ----------


@pytest.fixture
def heading_only_corpus(project_env):  # noqa: F811
    """One doc whose chunk carries a term (``Appendix``) only in its
    ``section_heading`` — never in the body text. Proves scan's MATCH is
    confined to the text column, while --heading-like can still reach it."""
    conn = open_db(project_env)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("hx", "memo.txt", "/tmp/memo.txt", 1, 100),
        )
        doc = conn.last_insert_rowid()
        ids = insert_document_chunks(conn, doc, [
            ChunkInput(text="The quarterly numbers are attached below.",
                       embedding=_emb(0.0), chunk_index=0,
                       section_heading="Appendix Tables", page_number=1),
        ])
    finally:
        conn.close()
    return {"project": project_env, "doc": doc, "chunk_id": ids[0]}


def test_scan_heading_only_term_yields_no_match(heading_only_corpus, capsys):
    """``Appendix`` lives only in the section_heading, so the text-qualified
    MATCH returns nothing — the snippet would never contain the term."""
    _run(heading_only_corpus, ["Appendix", "--match-terms"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["matches"] == []


def test_scan_body_term_still_matches(heading_only_corpus, capsys):
    """A term in the body text matches as before — text-only didn't break grep."""
    _run(heading_only_corpus, ["quarterly", "--match-terms"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 1
    assert [m["chunk_id"] for m in out["matches"]] == [heading_only_corpus["chunk_id"]]


def test_scan_heading_like_reaches_heading_only_term(heading_only_corpus, capsys):
    """Deliberate heading recall stays available: --heading-like surfaces the
    chunk a body-text MATCH would (correctly) miss."""
    _run(heading_only_corpus, ["quarterly", "--match-terms",
                               "--heading-like", "Appendix%"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 1
    assert [m["chunk_id"] for m in out["matches"]] == [heading_only_corpus["chunk_id"]]


def test_scan_unfiltered_omits_filters_object(scan_corpus, capsys):
    _run(scan_corpus, [MARKER])
    out = json.loads(capsys.readouterr().out)
    assert "filters" not in out


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
        "page_number", "section_heading", "content_type", "authored_date",
        "text", "text_length",
    }
    assert m["file_name"] == "filing_a.txt"
    assert m["document_id"] == scan_corpus["d1"]


# ---------- --count-by document (aggregate mode) ----------


def test_scan_count_by_document_histogram(scan_corpus, capsys):
    # MARKER hits d1 twice (chunks 1, 3) and d2 once: 3 chunks, 2 documents.
    _run(scan_corpus, [MARKER, "--count-by", "document"])
    out = json.loads(capsys.readouterr().out)
    assert out["count_by"] == "document"
    assert out["distinct_document_count"] == 2   # the headline
    assert out["total_chunk_count"] == 3         # the old `total`
    assert out["documents"] == [
        {"document_id": scan_corpus["d1"], "file_name": "filing_a.txt",
         "chunk_count": 2},
        {"document_id": scan_corpus["d2"], "file_name": "filing_b.txt",
         "chunk_count": 1},
    ]
    # The per-chunk projection's keys are gone in aggregate mode.
    for absent in ("matches", "preview", "total"):
        assert absent not in out


def test_scan_count_by_paginates_documents_with_stable_totals(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--count-by", "document", "--limit", "1"])
    p1 = json.loads(capsys.readouterr().out)
    assert [d["document_id"] for d in p1["documents"]] == [scan_corpus["d1"]]
    # Rollups are always the full, unpaginated totals.
    assert p1["distinct_document_count"] == 2
    assert p1["total_chunk_count"] == 3

    _run(scan_corpus, [MARKER, "--count-by", "document", "--limit", "1",
                       "--offset", "1"])
    p2 = json.loads(capsys.readouterr().out)
    assert [d["document_id"] for d in p2["documents"]] == [scan_corpus["d2"]]
    assert p2["distinct_document_count"] == 2


def test_scan_count_by_no_match_is_all_zeros(scan_corpus, capsys):
    _run(scan_corpus, ["phrase that appears nowhere", "--count-by", "document"])
    out = json.loads(capsys.readouterr().out)
    assert out["distinct_document_count"] == 0
    assert out["total_chunk_count"] == 0
    assert out["documents"] == []


def test_scan_count_by_rejects_preview_and_brief(scan_corpus, capsys):
    for bad in (["--preview", "100"], ["--brief"]):
        with pytest.raises(SystemExit) as exc:
            _run(scan_corpus, [MARKER, "--count-by", "document", *bad])
        # argparse-level conflict → the JSON usage envelope, exit 1 (issue #402).
        assert exc.value.code == 1
        assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


# ---------- --count-by '/regex/' (capture-group aggregate mode) ----------


BILL_MARKER = "Bill reference"
INCOME_MARKER = "Income reported"
BILL_RE = r"/H\.R\.\s*(\d+)/"
INCOME_RE = r"/reported:\s*\$([\d,]+)/"


@pytest.fixture
def templated_corpus(project_env):  # noqa: F811
    """Three filings with a templated bill-number field (one chunk repeats its
    bill, to prove per-match counting) and an income field on two of them."""
    conn = open_db(project_env)
    try:
        cur = conn.cursor()

        def _doc(file_hash, file_name):
            cur.execute(
                "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (file_hash, file_name, f"/tmp/{file_name}", 1, 50),
            )
            return conn.last_insert_rowid()

        d1, d2, d3 = _doc("ta", "f1.txt"), _doc("tb", "f2.txt"), _doc("tc", "f3.txt")
        insert_document_chunks(conn, d1, [
            ChunkInput(text="Bill reference: H.R. 4346 see also H.R. 4346",
                       embedding=_emb(0.0), chunk_index=0, page_number=1),
            ChunkInput(text="Income reported: $120,000 this period",
                       embedding=_emb(0.1), chunk_index=1, page_number=1),
        ])
        insert_document_chunks(conn, d2, [
            ChunkInput(text="Bill reference: H.R. 815", embedding=_emb(1.0),
                       chunk_index=0, page_number=1),
            ChunkInput(text="Income reported: $85,500 this period",
                       embedding=_emb(1.1), chunk_index=1, page_number=1),
        ])
        insert_document_chunks(conn, d3, [
            ChunkInput(text="Bill reference: H.R. 4346", embedding=_emb(2.0),
                       chunk_index=0, page_number=1),
            ChunkInput(text="Closing remarks only.", embedding=_emb(2.1),
                       chunk_index=1, page_number=1),
        ])
    finally:
        conn.close()
    return {"project": project_env, "d1": d1, "d2": d2, "d3": d3}


def test_scan_count_by_regex_buckets_per_match(templated_corpus, capsys):
    # 4346 hits twice in d1's chunk + once in d3 = 3; 815 once in d2.
    _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["count_by"] == BILL_RE          # the pattern is echoed verbatim
    assert out["distinct_value_count"] == 2    # the headline
    assert out["total_match_count"] == 4       # per-match, not per-chunk
    assert out["truncated"] is False
    assert out["groups"] == [
        {"value": "4346", "count": 3},
        {"value": "815", "count": 1},
    ]
    # Per-chunk and per-document projections are absent in this mode.
    for absent in ("matches", "documents", "total", "preview"):
        assert absent not in out


def test_scan_count_by_regex_sorts_count_desc_then_value(templated_corpus, capsys):
    # Income buckets tie at 1 each, so value-asc breaks the tie.
    _run(templated_corpus, [INCOME_MARKER, "--count-by", INCOME_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["groups"] == [
        {"value": "120,000", "count": 1},
        {"value": "85,500", "count": 1},
    ]


def test_scan_count_by_regex_paginates_with_full_rollups(templated_corpus, capsys):
    _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE, "--limit", "1"])
    p1 = json.loads(capsys.readouterr().out)
    assert p1["groups"] == [{"value": "4346", "count": 3}]
    assert p1["distinct_value_count"] == 2     # rollups stay full under pagination
    assert p1["total_match_count"] == 4

    _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE,
                            "--limit", "1", "--offset", "1"])
    p2 = json.loads(capsys.readouterr().out)
    assert p2["groups"] == [{"value": "815", "count": 1}]
    assert p2["distinct_value_count"] == 2


def test_scan_count_by_regex_respects_scope(templated_corpus, capsys):
    _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE,
                            "--in-documents", str(templated_corpus["d3"])])
    out = json.loads(capsys.readouterr().out)
    assert out["groups"] == [{"value": "4346", "count": 1}]
    assert out["total_match_count"] == 1
    assert out["filters"]["in_documents"] == [templated_corpus["d3"]]


def test_scan_count_by_regex_no_match_is_zeros(templated_corpus, capsys):
    _run(templated_corpus, ["phrase that appears nowhere", "--count-by", BILL_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["distinct_value_count"] == 0
    assert out["total_match_count"] == 0
    assert out["truncated"] is False
    assert out["groups"] == []


def test_scan_count_by_regex_no_capture_group_errors(templated_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(templated_corpus, [BILL_MARKER, "--count-by", r"/H\.R\.\s*\d+/"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "CAPTURE_NO_GROUP"


def test_scan_count_by_regex_uncompilable_errors(templated_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(templated_corpus, [BILL_MARKER, "--count-by", "/(/"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_CAPTURE_REGEX"


def test_scan_count_by_rejects_bare_non_document_value(templated_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(templated_corpus, [BILL_MARKER, "--count-by", "banana"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_COUNT_BY"


def test_scan_count_by_regex_rejects_preview_and_brief(templated_corpus, capsys):
    for bad in (["--preview", "100"], ["--brief"]):
        with pytest.raises(SystemExit) as exc:
            _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE, *bad])
        assert exc.value.code == 1
        assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


# ---------- --extract '/regex/' (capture-to-columns table, issue #420) ----------


EXTRACT_MARKER = "Filing entry"
# A named-group amount and a bare-group bill number on the same marker line.
AMOUNT_RE = r"/amount:\s*\$(?P<amount>[\d,]+)/"
BILL_NAMED_RE = r"/H\.R\.\s*(?P<bill>\d+)/"
BILL_BARE_RE = r"/H\.R\.\s*(\d+)/"


@pytest.fixture
def extract_corpus(project_env):  # noqa: F811
    """Filings sharing a marker, each carrying an amount; only some carry a bill.

    d1 has both amount + bill, d2 has only an amount (bill regex won't match →
    null cell), d3 carries a tag and its own amount. Lets us prove named +
    positional columns, the kept-row-with-null behaviour, and scope composition.
    """
    conn = open_db(project_env)
    try:
        cur = conn.cursor()

        def _doc(file_hash, file_name):
            cur.execute(
                "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (file_hash, file_name, f"/tmp/{file_name}", 1, 50),
            )
            return conn.last_insert_rowid()

        d1 = _doc("ea", "extract_a.txt")
        d2 = _doc("eb", "extract_b.txt")
        d3 = _doc("ec", "extract_c.txt")
        c1 = insert_document_chunks(conn, d1, [
            ChunkInput(text="Filing entry amount: $120,000 under H.R. 4346",
                       embedding=_emb(0.0), chunk_index=0, page_number=1),
        ])
        c2 = insert_document_chunks(conn, d2, [
            ChunkInput(text="Filing entry amount: $85,500 with no bill reference",
                       embedding=_emb(1.0), chunk_index=0, page_number=1),
        ])
        c3 = insert_document_chunks(conn, d3, [
            ChunkInput(text="Filing entry amount: $42,000 under H.R. 815",
                       embedding=_emb(2.0), chunk_index=0, page_number=1),
        ])

        # Tag d3 so we can prove --tag composes with --extract.
        cur.execute(
            "INSERT INTO tags (name, description) VALUES (?, ?)",
            ("senate", "senate filings"),
        )
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (d3, tag_id),
        )
    finally:
        conn.close()
    return {
        "project": project_env, "d1": d1, "d2": d2, "d3": d3,
        "c1": c1[0], "c2": c2[0], "c3": c3[0],
    }


def test_scan_extract_named_group_becomes_column(extract_corpus, capsys):
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["columns"] == ["amount"]
    assert out["total"] == 3
    by_doc = {r["document_id"]: r for r in out["rows"]}
    assert by_doc[extract_corpus["d1"]]["amount"] == "120,000"
    assert by_doc[extract_corpus["d2"]]["amount"] == "85,500"
    # Every row carries the locator trio plus the capture column.
    for r in out["rows"]:
        assert set(r) == {"chunk_id", "document_id", "file_name", "amount"}


def test_scan_extract_bare_group_gets_positional_name(extract_corpus, capsys):
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", BILL_BARE_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["columns"] == ["g1"]
    by_doc = {r["document_id"]: r for r in out["rows"]}
    assert by_doc[extract_corpus["d1"]]["g1"] == "4346"
    assert by_doc[extract_corpus["d3"]]["g1"] == "815"


def test_scan_extract_nonmatching_pattern_yields_null_keeps_row(extract_corpus, capsys):
    # d2 has an amount but no bill: its bill column is null, the row survives.
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE,
                          "--extract", BILL_NAMED_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["columns"] == ["amount", "bill"]
    assert out["total"] == 3
    by_doc = {r["document_id"]: r for r in out["rows"]}
    assert by_doc[extract_corpus["d2"]]["amount"] == "85,500"
    assert by_doc[extract_corpus["d2"]]["bill"] is None  # null, not dropped
    assert by_doc[extract_corpus["d1"]]["bill"] == "4346"


def test_scan_extract_column_collision_errors(extract_corpus, capsys):
    # Two bare groups both want g1 → ambiguous, rejected with a clean envelope.
    with pytest.raises(SystemExit) as exc:
        _run(extract_corpus, [EXTRACT_MARKER, "--extract", BILL_BARE_RE,
                              "--extract", r"/amount:\s*\$([\d,]+)/"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EXTRACT_COLUMN_COLLISION"
    assert out["column"] == "g1"


def test_scan_extract_composes_with_tag(extract_corpus, capsys):
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE,
                          "--tag", "senate"])
    out = json.loads(capsys.readouterr().out)
    # Only d3 carries the tag.
    assert out["total"] == 1
    assert [r["document_id"] for r in out["rows"]] == [extract_corpus["d3"]]
    assert out["rows"][0]["amount"] == "42,000"
    assert out["filters"]["tags"] == ["senate"]


def test_scan_extract_composes_with_file_like(extract_corpus, capsys):
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE,
                          "--file-like", "extract_a%"])
    out = json.loads(capsys.readouterr().out)
    assert [r["document_id"] for r in out["rows"]] == [extract_corpus["d1"]]
    assert out["rows"][0]["amount"] == "120,000"


def test_scan_extract_composes_with_in_documents(extract_corpus, capsys):
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE,
                          "--in-documents", f"{extract_corpus['d2']},{extract_corpus['d3']}"])
    out = json.loads(capsys.readouterr().out)
    got = sorted(r["document_id"] for r in out["rows"])
    assert got == sorted([extract_corpus["d2"], extract_corpus["d3"]])


def test_scan_extract_paginates_with_full_total(extract_corpus, capsys):
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE, "--limit", "1"])
    p1 = json.loads(capsys.readouterr().out)
    assert p1["total"] == 3          # honest total under pagination
    assert len(p1["rows"]) == 1
    assert p1["rows"][0]["document_id"] == extract_corpus["d1"]

    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE,
                          "--limit", "1", "--offset", "2"])
    p3 = json.loads(capsys.readouterr().out)
    assert p3["total"] == 3
    assert [r["document_id"] for r in p3["rows"]] == [extract_corpus["d3"]]


def test_scan_extract_no_match_is_empty_with_columns(extract_corpus, capsys):
    _run(extract_corpus, ["phrase that appears nowhere", "--extract", AMOUNT_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["rows"] == []
    assert out["columns"] == ["amount"]  # column set known even with no rows


def test_scan_extract_excludes_summary_chunks(extract_corpus, capsys):
    # scan is docs-only; an extract row is still a document chunk.
    _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE])
    out = json.loads(capsys.readouterr().out)
    assert all(isinstance(r["document_id"], int) for r in out["rows"])
    assert {r["document_id"] for r in out["rows"]} <= {
        extract_corpus["d1"], extract_corpus["d2"], extract_corpus["d3"],
    }


def test_scan_extract_invalid_regex_errors(extract_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(extract_corpus, [EXTRACT_MARKER, "--extract", "/(/"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_CAPTURE_REGEX"


def test_scan_extract_no_capture_group_errors(extract_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(extract_corpus, [EXTRACT_MARKER, "--extract", r"/H\.R\.\s*\d+/"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "CAPTURE_NO_GROUP"


def test_scan_extract_rejects_count_by(extract_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE,
                              "--count-by", "document"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


def test_scan_extract_rejects_preview_brief_returning(extract_corpus, capsys):
    for bad in (["--preview", "100"], ["--brief"], ["--returning", "chunk_id"]):
        with pytest.raises(SystemExit) as exc:
            _run(extract_corpus, [EXTRACT_MARKER, "--extract", AMOUNT_RE, *bad])
        assert exc.value.code == 1
        assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


def test_scan_count_by_regex_still_works_through_shared_machinery(templated_corpus, capsys):
    """--count-by '/regex/' is now extract-then-group-and-count over the same
    CaptureSpec; the histogram contract is unchanged."""
    _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE])
    out = json.loads(capsys.readouterr().out)
    assert out["groups"] == [
        {"value": "4346", "count": 3},
        {"value": "815", "count": 1},
    ]
    # A named bucket group still buckets the same (column name is irrelevant to
    # the histogram).
    _run(templated_corpus, [BILL_MARKER, "--count-by", r"/H\.R\.\s*(?P<bill>\d+)/"])
    named = json.loads(capsys.readouterr().out)
    assert named["groups"] == out["groups"]


# ---------- date scope (shared with list_documents / describe_corpus) ----------


def _date_d1(corpus, value="2024-03-01"):
    """Give d1's existing summary an authored_date (d2/d3 stay undated)."""
    conn = open_db(corpus["project"])
    try:
        conn.cursor().execute(
            "UPDATE summaries SET authored_date = ? WHERE document_id = ?",
            (value, corpus["d1"]),
        )
    finally:
        conn.close()


def test_scan_date_bound_filters_and_echoes(scan_corpus, capsys):
    _date_d1(scan_corpus)
    _run(scan_corpus, [MARKER, "--authored-after", "2024-01-01"])
    out = json.loads(capsys.readouterr().out)
    # Only d1 is dated and in-bounds, so only its marker chunks survive.
    assert out["total"] == 2
    assert all(m["document_id"] == scan_corpus["d1"] for m in out["matches"])
    f = out["filters"]
    assert f["authored_after"] == "2024-01-01"
    assert f["authored_before"] is None
    assert f["include_nulls"] is False
    # d2 and d3 have no summary → undated → dropped by the bound, counted.
    assert f["excluded_null_dated"] == 2


def test_scan_date_bound_includes_nulls(scan_corpus, capsys):
    _date_d1(scan_corpus)
    _run(scan_corpus, [MARKER, "--authored-after", "2024-01-01", "--include-nulls"])
    out = json.loads(capsys.readouterr().out)
    # Undated d2 rides along now, so its marker chunk returns too.
    assert out["total"] == 3
    assert out["filters"]["include_nulls"] is True
    assert out["filters"]["excluded_null_dated"] == 0


def test_scan_match_carries_authored_date(scan_corpus, capsys):
    """authored_date rides every match (default + brief), null for undated docs."""
    _date_d1(scan_corpus)  # d1 dated 2024-03-01; d2/d3 stay undated.
    for extra in ([], ["--brief"]):
        _run(scan_corpus, [MARKER, *extra])
        out = json.loads(capsys.readouterr().out)
        by_doc = {}
        for m in out["matches"]:
            assert "authored_date" in m  # present in every mode
            by_doc[m["document_id"]] = m["authored_date"]
        assert by_doc[scan_corpus["d1"]] == "2024-03-01"
        assert by_doc[scan_corpus["d2"]] is None  # undated → null


def test_scan_count_by_carries_date_filter(scan_corpus, capsys):
    _date_d1(scan_corpus)
    _run(scan_corpus, [MARKER, "--count-by", "document",
                       "--authored-after", "2024-01-01"])
    out = json.loads(capsys.readouterr().out)
    assert out["distinct_document_count"] == 1
    assert out["total_chunk_count"] == 2
    assert out["documents"] == [
        {"document_id": scan_corpus["d1"], "file_name": "filing_a.txt",
         "chunk_count": 2},
    ]
    assert out["filters"]["excluded_null_dated"] == 2


def test_scan_invalid_date_raises(scan_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(scan_corpus, [MARKER, "--authored-after", "2024"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INVALID_DATE"


# ---------- --sort {document,date} ----------


def _set_date(corpus, doc_key, value):
    """Stamp an authored_date onto a doc's summary, creating one if absent."""
    conn = open_db(corpus["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model, "
            "authored_date) VALUES (?, '', '', '', 'test', ?) "
            "ON CONFLICT(document_id) DO UPDATE SET authored_date = excluded.authored_date",
            (corpus[doc_key], value),
        )
    finally:
        conn.close()


def test_scan_sort_document_matches_default(scan_corpus, capsys):
    """--sort document is the explicit name for the default (document_id, chunk_index)."""
    _run(scan_corpus, [MARKER, "--sort", "document"])
    out = json.loads(capsys.readouterr().out)
    got = [(m["document_id"], m["chunk_index"]) for m in out["matches"]]
    assert got == [(scan_corpus["d1"], 1), (scan_corpus["d1"], 3), (scan_corpus["d2"], 0)]


def test_scan_sort_date_reorders_doc_groups_oldest_first(scan_corpus, capsys):
    # d2 is authored earlier than d1, so its group leads despite the higher id.
    _set_date(scan_corpus, "d1", "2024-03-01")
    _set_date(scan_corpus, "d2", "2020-01-01")
    _run(scan_corpus, [MARKER, "--sort", "date"])
    out = json.loads(capsys.readouterr().out)
    got = [(m["document_id"], m["chunk_index"]) for m in out["matches"]]
    # d2's group first (oldest), then d1's chunks in positional order.
    assert got == [(scan_corpus["d2"], 0), (scan_corpus["d1"], 1), (scan_corpus["d1"], 3)]


def test_scan_sort_date_puts_undated_last(scan_corpus, capsys):
    # Only d2 (the higher id) is dated; undated d1 must sort after it.
    _set_date(scan_corpus, "d2", "2020-01-01")
    _run(scan_corpus, [MARKER, "--sort", "date"])
    out = json.loads(capsys.readouterr().out)
    got = [(m["document_id"], m["chunk_index"]) for m in out["matches"]]
    assert got == [(scan_corpus["d2"], 0), (scan_corpus["d1"], 1), (scan_corpus["d1"], 3)]


def test_scan_sort_date_paginates_stably(scan_corpus, capsys):
    _set_date(scan_corpus, "d1", "2024-03-01")
    _set_date(scan_corpus, "d2", "2020-01-01")
    _run(scan_corpus, [MARKER, "--sort", "date", "--limit", "1", "--offset", "1"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 3
    # Page 2 of the chronological order: d1's first marker chunk.
    assert [(m["document_id"], m["chunk_index"]) for m in out["matches"]] == [
        (scan_corpus["d1"], 1),
    ]


def test_scan_count_by_sort_date_orders_histogram_chronologically(scan_corpus, capsys):
    _set_date(scan_corpus, "d1", "2024-03-01")
    _set_date(scan_corpus, "d2", "2020-01-01")
    _run(scan_corpus, [MARKER, "--count-by", "document", "--sort", "date"])
    out = json.loads(capsys.readouterr().out)
    # Default would be hit-count order (d1=2, d2=1); --sort date flips to oldest-first.
    assert [d["document_id"] for d in out["documents"]] == [
        scan_corpus["d2"], scan_corpus["d1"],
    ]


# ---------- --returning projection (issue #419) ----------


def test_scan_returning_projects_exact_fields(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--returning", "chunk_id,document_id"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 3  # envelope unchanged
    assert out["matches"]
    for m in out["matches"]:
        assert set(m) == {"chunk_id", "document_id"}


def test_scan_returning_respects_field_order(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--returning", "document_id,chunk_id"])
    out = json.loads(capsys.readouterr().out)
    # Dict order follows the requested order, not the whitelist order.
    assert list(out["matches"][0].keys()) == ["document_id", "chunk_id"]


def test_scan_returning_overrides_brief(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--returning", "text", "--brief"])
    out = json.loads(capsys.readouterr().out)
    # --returning wins over --brief's locator projection.
    for m in out["matches"]:
        assert set(m) == {"text"}


def test_scan_returning_unknown_field_errors(scan_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(scan_corpus, [MARKER, "--returning", "chunk_id,bogus"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"
    assert "chunk_id" in out["valid_fields"]
    assert "document_id" in out["valid_fields"]


def test_scan_default_projection_unchanged_without_returning(scan_corpus, capsys):
    _run(scan_corpus, [MARKER])
    out = json.loads(capsys.readouterr().out)
    m = out["matches"][0]
    assert set(m) == {
        "chunk_id", "document_id", "file_name", "chunk_index", "page_number",
        "section_heading", "content_type", "authored_date", "text", "text_length",
    }


def test_scan_count_by_document_chunk_id_selectable(scan_corpus, capsys):
    """The keystone: --count-by document rows expose a citable chunk_id via
    --returning, removing the re-read round-trip to recover an id."""
    _run(scan_corpus, [MARKER, "--count-by", "document", "--returning", "document_id,chunk_id"])
    out = json.loads(capsys.readouterr().out)
    assert out["distinct_document_count"] == 2  # aggregate envelope intact
    for d in out["documents"]:
        assert set(d) == {"document_id", "chunk_id"}
        # chunk_id is a real matching chunk in that document.
        assert isinstance(d["chunk_id"], int)


def test_scan_count_by_document_default_unchanged(scan_corpus, capsys):
    _run(scan_corpus, [MARKER, "--count-by", "document"])
    out = json.loads(capsys.readouterr().out)
    for d in out["documents"]:
        assert set(d) == {"document_id", "file_name", "chunk_count"}


def test_scan_count_by_document_returning_unknown_field_errors(scan_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(scan_corpus, [MARKER, "--count-by", "document", "--returning", "text"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    # 'text' is a per-chunk field, not in the count-by-document whitelist.
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"
    assert out["valid_fields"] == [
        "chunk_id", "document_id", "file_name", "chunk_count",
    ]


def test_scan_count_by_regex_rejects_returning(scan_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(scan_corpus, [MARKER, "--count-by", "/(ACME|BETA)/", "--returning", "chunk_id"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "RETURNING_NOT_APPLICABLE"
