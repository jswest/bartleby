"""Tests for the finding_annotations helpers (#731)."""

from __future__ import annotations

import pytest

from bartleby.db.annotations import (
    get_annotation,
    AnchorNotFound,
    count_annotations,
    delete_annotation,
    insert_annotation,
    list_annotations,
    locate_anchor,
    reparent_annotations,
    splits_protected_span,
)
from bartleby.db.connection import open_db
from tests._skill_fixtures import project_env, seed_finding, seeded_project  # noqa: F401

BODY = "alpha beta. gamma beta. delta beta"


def test_locate_single_occurrence():
    assert locate_anchor(BODY, "gamma") == (12, 17)


def test_locate_first_occurrence_by_default():
    assert locate_anchor(BODY, "beta") == (6, 10)


def test_locate_prefix_disambiguates():
    start, end = locate_anchor(BODY, "beta", prefix="gamma ")
    assert (start, end) == (18, 22)


def test_locate_suffix_disambiguates():
    assert locate_anchor(BODY, "beta", suffix="") == (6, 10)
    start, _ = locate_anchor(BODY, "beta", suffix=". gamma")
    assert start == 6
    start, _ = locate_anchor(BODY, "beta", suffix=". delta")
    assert start == 18


def test_locate_prefix_and_suffix():
    start, _ = locate_anchor(BODY, "beta", prefix="delta ", suffix="")
    assert start == 30
    assert locate_anchor(BODY, "beta", prefix="gamma ", suffix=". delta") == (18, 22)
    assert locate_anchor(BODY, "beta", prefix="alpha ", suffix=". delta") is None


def test_locate_prefix_longer_than_start_and_no_match():
    assert locate_anchor("beta", "beta", prefix="xxxx") is None
    assert locate_anchor(BODY, "missing") is None


def test_locate_empty_exact_invalid():
    with pytest.raises(ValueError):
        locate_anchor(BODY, "")


@pytest.fixture
def finding(seeded_project):  # noqa: F811
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES ('s1', 1)"
        )
        sid = conn.last_insert_rowid()
        fid, chunk_ids = seed_finding(conn, sid, body=BODY)
        yield conn, fid, chunk_ids[0]
    finally:
        conn.close()


def test_insert_list_roundtrip(finding):
    conn, fid, cid = finding
    whole = insert_annotation(conn, finding_id=fid, body="whole", is_human_author=True)
    span = insert_annotation(
        conn, finding_id=fid, body="span", is_human_author=False,
        anchor_exact="beta", anchor_prefix="gamma ", chunk_id=cid,
    )
    rows = list_annotations(conn, fid)
    assert [r["annotation_id"] for r in rows] == [whole, span]
    assert rows[0]["is_human_author"] == 1 and rows[0]["anchor_found"] is True
    assert rows[1]["is_human_author"] == 0 and rows[1]["chunk_id"] == cid
    assert rows[1]["anchor_found"] is True


def test_insert_validation(finding):
    conn, fid, _ = finding
    with pytest.raises(ValueError):
        insert_annotation(conn, finding_id=9999, body="x", is_human_author=True)
    with pytest.raises(AnchorNotFound):
        insert_annotation(conn, finding_id=fid, body="x", is_human_author=True,
                          anchor_exact="nope")
    with pytest.raises(ValueError):
        insert_annotation(conn, finding_id=fid, body="x", is_human_author=True,
                          chunk_id=999999)
    with pytest.raises(ValueError):
        insert_annotation(conn, finding_id=fid, body="x", is_human_author=True,
                          anchor_prefix="a")
    assert list_annotations(conn, fid) == []


def test_stale_anchor_reports_not_found(finding):
    conn, fid, _ = finding
    insert_annotation(conn, finding_id=fid, body="x", is_human_author=True,
                      anchor_exact="gamma")
    conn.cursor().execute(
        "UPDATE findings SET body = 'rewritten' WHERE finding_id = ?", (fid,)
    )
    assert list_annotations(conn, fid)[0]["anchor_found"] is False


def test_delete_and_count(finding):
    conn, fid, _ = finding
    a = insert_annotation(conn, finding_id=fid, body="a", is_human_author=True)
    insert_annotation(conn, finding_id=fid, body="b", is_human_author=True)
    assert count_annotations(conn, [fid, 12345]) == {fid: 2, 12345: 0}
    assert count_annotations(conn, []) == {}
    assert delete_annotation(conn, a) is True
    assert delete_annotation(conn, a) is False
    assert count_annotations(conn, [fid]) == {fid: 1}


def test_finding_delete_cascades(finding):
    conn, fid, _ = finding
    insert_annotation(conn, finding_id=fid, body="a", is_human_author=True)
    conn.cursor().execute("DELETE FROM findings WHERE finding_id = ?", (fid,))
    assert count_annotations(conn, [fid]) == {fid: 0}


def test_get_annotation_by_id(finding):
    conn, finding_id, _ = finding
    aid = insert_annotation(
        conn, finding_id=finding_id, body="note", is_human_author=True,
        anchor_exact="gamma",
    )
    row = get_annotation(conn, aid)
    assert row is not None
    assert row["annotation_id"] == aid
    assert row["finding_id"] == finding_id
    assert row["body"] == "note"
    assert row["anchor_found"] is True
    assert get_annotation(conn, aid + 1000) is None


MARKED = "Revenue rose [^chunk:12] sharply, see [the memo](http://x/y)."


def _span(text: str) -> tuple[int, int]:
    start = MARKED.index(text)
    return start, start + len(text)


@pytest.mark.parametrize("quote", [
    "rose [^chunk:1",       # end inside [^chunk:12]
    "chunk:12] sharply",    # start inside [^chunk:12]
    "memo](http",           # end inside ](http://x/y)
    "x/y).",                # start inside ](http://x/y)
])
def test_splits_protected_span_rejects_split_markers(quote):
    assert splits_protected_span(MARKED, *_span(quote)) is True


@pytest.mark.parametrize("quote", [
    "rose [^chunk:12] sharply",  # wholly contains a marker
    "[^chunk:12]",               # exactly a marker (boundaries on its edges)
    "Revenue rose",              # plain prose
    "the memo](http://x/y)",     # wholly contains a link target
])
def test_splits_protected_span_accepts_whole_markers(quote):
    assert splits_protected_span(MARKED, *_span(quote)) is False


def test_insert_rejects_anchor_splitting_a_marker(seeded_project):  # noqa: F811
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES ('s1', 1)"
        )
        fid, _ = seed_finding(conn, conn.last_insert_rowid(), body=MARKED)
        for bad in ("rose [^chunk:1", "memo](http"):
            with pytest.raises(AnchorNotFound, match="citation marker or link"):
                insert_annotation(conn, finding_id=fid, body="x",
                                  is_human_author=True, anchor_exact=bad)
        assert list_annotations(conn, fid) == []
        insert_annotation(conn, finding_id=fid, body="ok", is_human_author=True,
                          anchor_exact="rose [^chunk:12] sharply")
        assert len(list_annotations(conn, fid)) == 1
    finally:
        conn.close()


def test_reparent_annotations_moves_notes(finding):
    conn, fid, _ = finding
    sid = conn.cursor().execute("SELECT session_id FROM findings").fetchone()[0]
    other, _ = seed_finding(conn, sid, body="other body")
    insert_annotation(conn, finding_id=fid, body="a", is_human_author=True,
                      anchor_exact="gamma")
    insert_annotation(conn, finding_id=fid, body="b", is_human_author=False)
    assert reparent_annotations(conn, [fid], other) == 2
    assert reparent_annotations(conn, [], other) == 0
    rows = list_annotations(conn, other)
    assert [r["body"] for r in rows] == ["a", "b"]
    assert rows[0]["anchor_found"] is False  # quoted the old body
    assert count_annotations(conn, [fid]) == {fid: 0}


def test_walled_session_hides_foreign_agent_notes(finding):
    conn, fid, _ = finding
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (name, memory_enabled) VALUES ('me', 0)")
    me = conn.last_insert_rowid()
    cur.execute("INSERT INTO sessions (name, memory_enabled) VALUES ('them', 1)")
    them = conn.last_insert_rowid()
    insert_annotation(conn, finding_id=fid, body="human", is_human_author=True)
    insert_annotation(conn, finding_id=fid, body="mine", is_human_author=False,
                      session_id=me)
    insert_annotation(conn, finding_id=fid, body="theirs", is_human_author=False,
                      session_id=them)
    walled = list_annotations(conn, fid, walled_session_id=me)
    assert [r["body"] for r in walled] == ["human", "mine"]
    assert len(list_annotations(conn, fid)) == 3
    assert count_annotations(conn, [fid], walled_session_id=me) == {fid: 2}
