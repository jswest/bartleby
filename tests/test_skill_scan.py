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
        assert set(m) == {"document_id", "file_name", "chunk_id", "page_number"}


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
        "page_number", "section_heading", "content_type", "text", "text_length",
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


def test_scan_count_by_rejects_preview_and_brief(scan_corpus):
    for bad in (["--preview", "100"], ["--brief"]):
        with pytest.raises(SystemExit) as exc:
            _run(scan_corpus, [MARKER, "--count-by", "document", *bad])
        # argparse-level conflict → exit 2.
        assert exc.value.code == 2


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
    assert json.loads(capsys.readouterr().out)["code"] == "COUNT_BY_NO_CAPTURE"


def test_scan_count_by_regex_uncompilable_errors(templated_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(templated_corpus, [BILL_MARKER, "--count-by", "/(/"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_COUNT_BY_REGEX"


def test_scan_count_by_rejects_bare_non_document_value(templated_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(templated_corpus, [BILL_MARKER, "--count-by", "banana"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_COUNT_BY"


def test_scan_count_by_regex_rejects_preview_and_brief(templated_corpus):
    for bad in (["--preview", "100"], ["--brief"]):
        with pytest.raises(SystemExit) as exc:
            _run(templated_corpus, [BILL_MARKER, "--count-by", BILL_RE, *bad])
        assert exc.value.code == 2


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
