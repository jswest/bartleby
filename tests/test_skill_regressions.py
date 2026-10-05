"""Regression tests for real past bugs in the skill scripts. Each section names
the issue it guards.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading

import pytest

from bartleby.db.chunks import ChunkInput
from bartleby.db.chunks import insert_document_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.skill_scripts import edit_finding
from bartleby.skill_scripts import list_documents
from bartleby.skill_scripts import read_chunks
from bartleby.skill_scripts import save_finding
from bartleby.skill_scripts import save_summary
from bartleby.skill_scripts import scan
from bartleby.skill_scripts import search
from bartleby.skill_scripts._common import SkillError
from bartleby.skill_scripts._common import reject_malformed_citations
import bartleby.project
import bartleby.session as session_mod

from tests._skill_fixtures import _emb
from tests._skill_fixtures import mock_embed  # noqa: F401
from tests._skill_fixtures import project_env  # noqa: F401
from tests._skill_fixtures import seeded_project  # noqa: F401
from tests._skill_fixtures import stub_embed  # noqa: F401


# ---------- search: #55 scoped semantic search, #465 source deleted
# mid-search, #472 --returning typo on zero hits ----------


@pytest.mark.usefixtures("stub_embed")
def test_search_semantic_scope_survives_global_nearest_elsewhere(
    seeded_project, capsys, monkeypatch
):
    """Regression for #55: when more out-of-scope chunks than the over-fetch
    window are nearer to the query than the in-scope chunks, a semantic search
    scoped to doc_a must still return doc_a's chunks. The old post-filter
    (fetch k globally-nearest, then drop out-of-scope rows) starved to [] here;
    pushing the scope into the index fixes it."""
    # Seed a decoy document with MORE near-query chunks than the over-fetch
    # window, so every chunk in the global top-`overfetch` is out of scope and a
    # post-filter to doc_a would wipe the result set to empty.
    limit = 1  # mirror work()'s overfetch calc for this query's --limit
    overfetch = max(
        limit * search.OVERFETCH_MULTIPLIER, search.OVERFETCH_FLOOR
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
    monkeypatch.setattr(search, "_embed_query", lambda q: near_decoys)

    # Premise: unscoped, the global nearest hit is NOT in doc_a.
    search.main([
        "--project", seeded_project["project"],
        "--semantic", "--documents", "--limit", str(limit),
        "anything",
    ])
    unscoped = json.loads(capsys.readouterr().out)
    assert unscoped["results"][0]["source_id"] != f"document:{seeded_project['doc_a']}"

    # Scoped to doc_a, the in-scope chunks must come back — not [].
    search.main([
        "--project", seeded_project["project"],
        "--semantic", "--documents", "--limit", str(limit),
        "--in-documents", f"document:{seeded_project['doc_a']}",
        "anything",
    ])
    scoped = json.loads(capsys.readouterr().out)
    assert scoped["results"], "scoped semantic search starved to empty (issue #55)"
    for r in scoped["results"]:
        assert r["source_kind"] == "document"
        assert r["source_id"] == f"document:{seeded_project['doc_a']}"


@pytest.mark.usefixtures("stub_embed")
def test_search_returning_unknown_field_errors_on_zero_hits(seeded_project, capsys):
    """A typo'd --returning must error even when the query matches nothing,
    rather than coming back as a silent empty result set."""
    with pytest.raises(SystemExit) as exc:
        search.main([
            "--project", seeded_project["project"], "--full-text",
            "zzzznomatchzzz", "--returning", "chunk_id,nope",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


@pytest.mark.usefixtures("stub_embed")
def test_search_completes_when_source_deleted_underneath(seeded_project, capsys):
    """A chunk whose (source_kind, source_id) pair no longer resolves — its
    source row deleted by a concurrent session between fetch and name resolution
    — must degrade that one hit's source_name to "" rather than aborting the
    whole search with KeyError → INTERNAL_ERROR (issue #465)."""
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
        search.main([
            "--project", seeded_project["project"],
            "--full-text", "ghostword", *extra,
        ])
        out = json.loads(capsys.readouterr().out)
        assert "error" not in out, f"search aborted with {extra}: {out}"
        hit = next(r for r in out["results"] if r["chunk_id"])
        assert hit["source_name"] == ""


# ---------- read_chunks: #472 --returning typo on zero rows ----------


def test_read_chunks_returning_unknown_field_errors_on_zero_rows(
    seeded_project, capsys
):
    """A typo'd --returning must error even when the read resolves to no rows,
    rather than coming back as a silent empty result. A non-existent document id
    reaches work() (parse succeeds), so the up-front whitelist check fires."""
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document-id", "document:999999",
            "--returning", "chunk_id,bogus",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


# ---------- list_documents: #472 --returning typo on zero documents ----------


def test_list_documents_returning_unknown_field_errors_on_zero_documents(
    seeded_project, capsys
):
    """A typo'd --returning must error even when the filter matches no documents,
    rather than coming back as a silent empty list."""
    with pytest.raises(SystemExit) as exc:
        list_documents.main([
            "--project", seeded_project["project"],
            "--file-like", "zzzznomatchzzz",
            "--returning", "document_id,bogus",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


# ---------- scan: #653 --body-matches + --count-by, #472 --returning typo on zero matches ----------


def test_scan_returning_unknown_field_errors_on_zero_matches(seeded_project, capsys):
    """The whitelist check must fire even when the query matches no rows — else a
    typo'd field reads as an empty result instead of a broken flag."""
    with pytest.raises(SystemExit) as exc:
        scan.main(["--project", seeded_project["project"], "zzzznomatchzzz",
                   "--returning", "chunk_id,bogus"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


def test_scan_count_by_document_returning_unknown_field_errors_on_zero_matches(
    seeded_project, capsys
):
    with pytest.raises(SystemExit) as exc:
        scan.main(["--project", seeded_project["project"], "zzzznomatchzzz",
                   "--count-by", "document", "--returning", "text"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


def test_scan_body_matches_rejects_count_by(seeded_project, capsys):
    """--body-matches filters per-chunk matches; --count-by returns an aggregate
    histogram that can't carry that filter, so the combination is rejected rather
    than silently ignored (the #653 × #641 seam)."""
    with pytest.raises(SystemExit) as exc:
        scan.main(["--project", seeded_project["project"], "alpha",
                   "--body-matches", "/GAMMA/", "--count-by", "document"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


# ---------- save_summary: #467 authored_date carry-forward ----------


def test_save_summary_replace_carries_authored_date_forward(
    seeded_project, capsys
):
    # Save a dated summary, then re-save it WITHOUT --authored-date. The prior
    # date must survive the full-row replace rather than being nulled (issue
    # #467) — otherwise the re-save silently drops the doc out of every
    # authored-date scope.
    doc_b = seeded_project["doc_b"]
    save_summary.main([
        "--project", seeded_project["project"],
        "--document-id", f"document:{doc_b}",
        "--title", "Beta dated",
        "--description", "First save, with a date.",
        "--text", "Dated body.",
        "--authored-date", "2024-09-12",
    ])
    capsys.readouterr()

    # Re-save with no --authored-date at all.
    save_summary.main([
        "--project", seeded_project["project"],
        "--document-id", f"document:{doc_b}",
        "--title", "Beta v2",
        "--description", "Second save, no date arg.",
        "--text", "Replacement body.",
    ])
    capsys.readouterr()

    conn = open_db(seeded_project["project"])
    try:
        row = conn.cursor().execute(
            "SELECT title, authored_date FROM summaries WHERE document_id = ?",
            (doc_b,),
        ).fetchone()
        assert row == ("Beta v2", "2024-09-12")
    finally:
        conn.close()


# ---------- save_finding: #698 bracketed-year citations ----------


def test_malformed_check_exempts_bracketed_year_neutral_citation():
    """A caret-less ``[YYYY]``-shaped neutral case citation (e.g.
    ``Re Estate of Example [1998] HKLRD 771``) is ordinary prose, not a
    citation-shaped marker — must not trip MALFORMED_CITATION. Regression
    for #698."""
    body = "Corpus claim[^chunk:42]. See Re Estate of Example [1998] HKLRD 771."
    reject_malformed_citations(body)  # must not raise


def test_malformed_check_still_rejects_non_four_digit_bracket():
    """The exemption is narrowly the 4-digit year shape — a genuinely
    malformed short marker like ``[12]`` is still rejected."""
    with pytest.raises(SkillError) as exc:
        reject_malformed_citations("Corpus claim[^chunk:42]. Bad cite [12] here.")
    assert exc.value.code == "MALFORMED_CITATION"
    assert "[12]" in exc.value.extra["malformed_markers"]


def test_malformed_check_still_rejects_careted_four_digit_bracket():
    """The 4-digit exemption is caret-less only: ``[^1998]`` is still the
    obsolete bare-chunk-citation shape, not a case citation (those never
    carry a caret) — still rejected."""
    with pytest.raises(SkillError) as exc:
        reject_malformed_citations("Corpus claim[^chunk:42]. Bad cite [^1998] here.")
    assert exc.value.code == "MALFORMED_CITATION"
    assert "[^1998]" in exc.value.extra["malformed_markers"]


# ---------- #562 concurrent-write BusyError ----------


class _ThreadLocalStdout:
    """Route ``sys.stdout`` writes to a per-thread buffer.

    The skill scripts print their JSON envelope to ``sys.stdout`` via
    ``_print_json``. With many writer threads sharing the one process stdout,
    their envelopes would interleave into an unparseable mush. This proxy hands
    each thread its own ``StringIO`` so every thread reads back exactly its own
    invocation's result; threads without a registered buffer fall through to the
    real stream.
    """

    def __init__(self, real):
        self._real = real
        self._buffers: dict[int, object] = {}

    def register(self, buf) -> None:
        self._buffers[threading.get_ident()] = buf

    def _target(self):
        return self._buffers.get(threading.get_ident(), self._real)

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        return self._target().flush()


def _run_writer(fn, argv, results, idx, tls):
    import io

    buf = io.StringIO()
    tls.register(buf)
    outcome: dict = {"idx": idx}
    try:
        fn(argv)
    except SystemExit as e:
        # The runner exits non-zero on any error (BusyError included).
        outcome["exit_code"] = e.code
    except BaseException as e:  # noqa: BLE001 — surface anything unexpected
        outcome["raised"] = f"{type(e).__name__}: {e}"
    finally:
        outcome["stdout"] = buf.getvalue()
    results[idx] = outcome


def _cited_chunk(project, doc_id):
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (doc_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def _assert_no_busy(outcomes):
    """No writer may fail — and emphatically not with a BusyError."""
    for o in outcomes:
        # An unexpected non-SystemExit escape is a hard failure.
        assert "raised" not in o, f"writer {o['idx']} raised: {o.get('raised')}"
        stdout = o.get("stdout", "")
        envelope = json.loads(stdout) if stdout.strip() else {}
        code = envelope.get("code")
        err = envelope.get("error", "")
        assert "Busy" not in str(code) and "Busy" not in str(err), (
            f"writer {o['idx']} hit a BusyError: {envelope}"
        )
        # Success: the runner exited 0 (no SystemExit recorded) and the
        # envelope carries no error code.
        assert o.get("exit_code") is None, (
            f"writer {o['idx']} exited non-zero: {envelope}"
        )
        assert code is None, f"writer {o['idx']} reported error {code}: {envelope}"


def _run_concurrently(fn, argvs):
    """Run ``fn(argv)`` for each argv on its own thread; return ordered outcomes.

    Swaps in a thread-local stdout proxy for the duration (restored in a
    ``finally`` so the autouse ``BARTLEBY_HOME`` isolation is never touched —
    ``monkeypatch.undo()`` would revert *every* patch, including that one).
    """
    tls = _ThreadLocalStdout(sys.stdout)
    real_stdout = sys.stdout
    sys.stdout = tls
    try:
        results: dict[int, dict] = {}
        threads = [
            threading.Thread(target=_run_writer, args=(fn, argv, results, i, tls))
            for i, argv in enumerate(argvs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        sys.stdout = real_stdout
    return [results[i] for i in range(len(argvs))]


def _save_one(project, title, body_file) -> int:
    """Save a single finding sequentially; return its id (capturing stdout)."""
    import io

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        save_finding.main([
            "--project", project,
            "--title", title,
            "--description", "seed",
            "--body-file", str(body_file),
        ])
    finally:
        sys.stdout = old
    # finding_id is now a type-tagged "finding:<id>"; return the bare int.
    return int(json.loads(buf.getvalue())["finding_id"].split(":", 1)[1])


def test_concurrent_save_findings_serialize(seeded_project, tmp_path):
    """N threads saving findings into one session DB all succeed (serialize)."""
    project = seeded_project["project"]
    cited = _cited_chunk(project, seeded_project["doc_a"])

    n = 6
    argvs = []
    for i in range(n):
        bf = tmp_path / f"save_{i}.md"
        bf.write_text(f"Concurrent claim {i}[^chunk:{cited}].", encoding="utf-8")
        argvs.append([
            "--project", project,
            "--title", f"concurrent-{i}",
            "--description", "race",
            "--body-file", str(bf),
        ])

    outcomes = _run_concurrently(save_finding.main, argvs)
    _assert_no_busy(outcomes)

    # All N findings landed — serialization preserved every write.
    conn = open_db(project)
    try:
        count = conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE title LIKE 'concurrent-%'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == n


def test_concurrent_edit_findings_serialize(seeded_project, tmp_path):
    """N threads editing distinct findings in one session DB all succeed.

    ``edit_finding`` reads the stored body before it rewrites it — the exact
    read-then-write shape that tripped the deferred-transaction BUSY trap.
    """
    project = seeded_project["project"]
    cited = _cited_chunk(project, seeded_project["doc_a"])

    n = 6
    finding_ids: list[int] = []
    # Seed N findings up front (sequentially) so each thread edits its own.
    for i in range(n):
        bf = tmp_path / f"seed_{i}.md"
        bf.write_text(f"Seed body {i}[^chunk:{cited}].", encoding="utf-8")
        finding_ids.append(_save_one(project, f"edit-seed-{i}", bf))

    argvs = []
    for i in range(n):
        bf = tmp_path / f"edit_{i}.md"
        bf.write_text(f"Edited body {i}[^chunk:{cited}].", encoding="utf-8")
        argvs.append([
            "--project", project,
            "--finding-id", f"finding:{finding_ids[i]}",
            "--body-file", str(bf),
        ])

    outcomes = _run_concurrently(edit_finding.main, argvs)
    _assert_no_busy(outcomes)

    # Every edit took effect.
    conn = open_db(project)
    try:
        cur = conn.cursor()
        for i, fid in enumerate(finding_ids):
            body = cur.execute(
                "SELECT body FROM findings WHERE finding_id = ?", (fid,)
            ).fetchone()[0]
            assert body == f"Edited body {i}[^chunk:{cited}]."
    finally:
        conn.close()


# ---------- #632 --file-like SQL-variable crash ----------


_HIGHCARD_COUNT = 33_000


_PATTERN = "hc_doc_%"


_MARKER = "highcard unique marker phrase"


@pytest.fixture
def highcard_project(project_env):  # noqa: F811
    """A project with 33,000 matching documents and 1 non-matching document.

    All 33k documents share a filename prefix ``hc_doc_`` and contain a unique
    marker phrase. One extra document (``other_doc.txt``) does NOT match the
    pattern, to verify the scope excludes it.
    """
    conn = open_db(project_env)
    try:
        cur = conn.cursor()

        # Bulk-insert 33k documents without chunks — we only need them for the
        # scope/filter path, not for FTS matches. One chunk per doc keeps the
        # test honest for scan (FTS must find something) without being slow.
        rows = [
            (f"h{i:07d}", f"hc_doc_{i:07d}.txt", f"/tmp/hc_doc_{i:07d}.txt", 1, 50)
            for i in range(_HIGHCARD_COUNT)
        ]
        cur.executemany(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        # Fetch the first inserted id to compute the range.
        first_id = conn.last_insert_rowid() - _HIGHCARD_COUNT + 1
        doc_ids = list(range(first_id, first_id + _HIGHCARD_COUNT))

        # One chunk per doc — content contains the marker so scan can find it.
        insert_document_chunks(conn, doc_ids[0], [
            ChunkInput(
                text=_MARKER,
                embedding=_emb(0.0),
                chunk_index=0,
            ),
        ])

        # One document that does NOT match the pattern — control for exclusion.
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("other_hash", "other_doc.txt", "/tmp/other_doc.txt", 1, 20),
        )
        other_id = conn.last_insert_rowid()
    finally:
        conn.close()

    return {
        "project": project_env,
        "doc_ids": doc_ids,
        "other_id": other_id,
    }


def test_file_like_highcard_list_documents(highcard_project, capsys):
    """list_documents --file-like with >32,766 matches must not crash, and the
    non-matching document must not appear."""
    list_documents.main([
        "--project", highcard_project["project"],
        "--file-like", _PATTERN, "--limit", str(_HIGHCARD_COUNT + 1),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == _HIGHCARD_COUNT
    assert len(out["documents"]) == _HIGHCARD_COUNT
    assert out["filters"]["file_like"] == [_PATTERN]
    returned_ids = [int(d["id"].split(":")[1]) for d in out["documents"]]
    assert highcard_project["other_id"] not in returned_ids


def test_file_like_highcard_scan(highcard_project, capsys):
    """scan --file-like with >32,766 matches must not crash."""
    scan.main([
        "--project", highcard_project["project"],
        _MARKER, "--file-like", _PATTERN, "--limit", "5",
    ])
    out = json.loads(capsys.readouterr().out)
    # Exactly one doc has the marker chunk — the scope must not widen.
    assert out["total"] == 1
    assert out["filters"]["file_like"] == [_PATTERN]


def test_file_like_highcard_search(highcard_project, capsys):
    """search --file-like with >32,766 matches must not crash."""
    search.main([
        "--project", highcard_project["project"],
        _MARKER, "--file-like", _PATTERN, "--limit", "5", "--full-text",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == [_PATTERN]
    # Exactly one doc has the marker chunk; the non-matching doc must be absent.
    assert len(out["results"]) >= 1
    result_source_ids = [r["source_id"] for r in out["results"]]
    assert f"document:{highcard_project['other_id']}" not in result_source_ids


# ---------- write_active_session_id: #501 atomic publish, #553 concurrent-writer race ----------


@pytest.fixture
def project():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    bartleby.project.create_project("alpha")
    return "alpha"


def test_write_active_session_id_is_atomic_no_partial_read(project, monkeypatch):
    # write_active_session_id must publish via an atomic rename: a concurrent
    # reader interleaved mid-write sees either the old id or the new one in
    # full, never a truncated line. We intercept os.replace to run a read at
    # exactly the moment the new content exists in a temp file but the pointer
    # hasn't been swapped yet.
    session_mod.write_active_session_id(project, 11)

    observed = []
    orig_replace = os.replace

    def spy_replace(src, dst):
        # Mid-write: reader still sees the OLD committed value, never a partial.
        observed.append(session_mod.read_active_session_id(project))
        orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    session_mod.write_active_session_id(project, 22)

    assert observed == [11]  # old value still visible until the rename lands
    assert session_mod.read_active_session_id(project) == 22


def test_write_active_session_id_concurrent_writers_no_race(project):
    # Regression for #553: a shared ".active_session.tmp" let concurrent writers
    # clobber one temp and race on the rename — the second os.replace raised
    # FileNotFoundError. Unique temps fix it: many overlapping writers all
    # succeed, the pointer lands on one written id, and no temp is left behind.
    import threading

    ids = list(range(1, 51))
    errors: list[Exception] = []

    def writer(sid: int) -> None:
        try:
            for _ in range(10):
                session_mod.write_active_session_id(project, sid)
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(sid,)) for sid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert session_mod.read_active_session_id(project) in ids
    pdir = bartleby.project.get_project_dir(project)
    assert not list(pdir.glob(".active_session.*.tmp"))
