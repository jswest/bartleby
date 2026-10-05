"""Memory wall: a memory_enabled=0 session cannot see or touch other sessions'
findings or agent notes (ARCHITECTURE.md invariant).
"""

from __future__ import annotations

import json
import struct

import pytest

from bartleby.db.annotations import count_annotations
from bartleby.db.annotations import insert_annotation
from bartleby.db.annotations import list_annotations
from bartleby.db.chunks import ChunkInput
from bartleby.db.chunks import insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.session import start_session
from bartleby.skill_scripts import annotate_finding
from bartleby.skill_scripts import delete_annotation
from bartleby.skill_scripts import delete_finding
from bartleby.skill_scripts import edit_finding
from bartleby.skill_scripts import list_findings
from bartleby.skill_scripts import merge_findings
from bartleby.skill_scripts import read_chunks
from bartleby.skill_scripts import read_finding
from bartleby.skill_scripts import save_finding
from bartleby.skill_scripts import search as search_script

from tests._skill_fixtures import assert_chunk_tables_consistent  # noqa: F401
from tests._skill_fixtures import dated_corpus  # noqa: F401
from tests._skill_fixtures import mock_embed  # noqa: F401
from tests._skill_fixtures import project_env  # noqa: F401
from tests._skill_fixtures import seed_finding  # noqa: F401
from tests._skill_fixtures import seed_finding_via_main  # noqa: F401
from tests._skill_fixtures import seeded_project  # noqa: F401
from tests._skill_fixtures import unprefix  # noqa: F401


# ---------- skill_search ----------


@pytest.fixture
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


@pytest.mark.usefixtures("stub_embed")
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


# ---------- skill_list_findings ----------


def test_list_findings_memory_off_scopes_to_own_session(seeded_project, capsys):
    """Memory-off lists only the active session's findings, hiding others'."""
    from bartleby.session import start_session

    project = seeded_project["project"]
    # A finding authored by some *other* session.
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        seed_finding(conn, session_id=author, title="hidden", description="x")
    finally:
        conn.close()

    # Start a memory-off session and give it one finding of its own.
    info = start_session(project, memory_enabled=False)
    conn = open_db(project)
    try:
        seed_finding(conn, session_id=info["session_id"],
                     title="mine", description="y")
    finally:
        conn.close()

    list_findings.main(["--project", project])
    out = json.loads(capsys.readouterr().out)
    # Only this session's finding is visible; the other session's is excluded,
    # and total reflects the scoped set.
    assert out["total"] == 1
    assert [f["title"] for f in out["findings"]] == ["mine"]


# ---------- skill_read_chunks ----------


def _seed_finding_chunk(conn, *, session_id: int, body: str = "finding body") -> int:
    """Seed a foreign-walled finding and return its body chunk_id.

    Title is fixed at ``"secret finding"`` because the memory-wall leak tests
    assert that exact string never appears in a walled response — a different
    title would make those assertions vacuous.
    """
    _, [chunk_id] = seed_finding(
        conn, session_id=session_id, title="secret finding", body=body,
    )
    return chunk_id


def _other_session(conn, name: str = "author") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)", (name, 1),
    )
    return conn.last_insert_rowid()


def test_read_chunks_memory_off_drops_foreign_finding_chunk(seeded_project, capsys):
    """A memory-off session sees a foreign finding chunk only as missing — no leak."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        author = _other_session(conn)
        foreign = _seed_finding_chunk(
            conn, session_id=author, body="confidential prior conclusion",
        )
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    read_chunks.main(["--project", project, "--chunks", f"chunk:{foreign}"])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == [f"chunk:{foreign}"]
    assert out["chunks"] == []
    # The body text and finding title must not appear anywhere in the response.
    assert "confidential prior conclusion" not in json.dumps(out)
    assert "secret finding" not in json.dumps(out)


def test_read_chunks_around_memory_off_foreign_finding_walled(seeded_project, capsys):
    """--around-chunk on a foreign finding chunk raises MEMORY_OFF."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        author = _other_session(conn)
        foreign = _seed_finding_chunk(conn, session_id=author)
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    with pytest.raises(SystemExit) as exc:
        read_chunks.main(["--project", project, "--around-chunk", f"chunk:{foreign}"])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"


# ---------- skill_read_finding ----------


def _run_read_finding(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        read_finding.main(argv)
    return exc.value.code, capsys.readouterr()


def test_read_finding_memory_off_other_session(seeded_project, capsys):
    """A memory-off session cannot read a finding authored by another session."""
    from bartleby.session import start_session

    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        finding_id, _ = seed_finding(conn, session_id=author)
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    code, captured = _run_read_finding(capsys, [
        "--project", project, "--finding-id", f"finding:{finding_id}",
    ])
    assert code == 1
    out = json.loads(captured.out)
    assert out["code"] == "MEMORY_OFF"


# ---------- skill_edit_finding ----------


def _seed_finding(seeded_project, tmp_path, capsys, *, body_suffix: str = "") -> dict:
    """Seed the baseline finding these edit tests assert against ("Original …")."""
    return seed_finding_via_main(
        seeded_project, tmp_path, capsys,
        title="Original title", description="Original description.",
        body_suffix=body_suffix,
    )


def test_edit_finding_memory_off_other_session(seeded_project, tmp_path, capsys):
    """A memory-off session cannot edit (and thereby read back) a finding
    authored by another session — the response echoes the body, so an ungated
    --title-only edit would be a read-by-write bypass of the memory wall."""
    from bartleby.session import start_session

    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]
    fid = unprefix(finding_id)

    # The seed finding belongs to the (memory-on) default session; open a
    # fresh memory-off session as the would-be attacker.
    start_session(seeded_project["project"], memory_enabled=False)

    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", finding_id,
            "--title", "Hijacked title",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"
    # No body or citation content leaks back through the error.
    assert "body" not in out
    assert "citations" not in out

    # And the foreign finding was not mutated.
    conn = open_db(seeded_project["project"])
    try:
        title = conn.cursor().execute(
            "SELECT title FROM findings WHERE finding_id = ?",
            (fid,),
        ).fetchone()[0]
        assert title == "Original title"
    finally:
        conn.close()


# ---------- skill_merge_findings ----------


def _doc_chunk_ids(project, doc_id) -> list[int]:
    conn = open_db(project)
    try:
        return [
            r[0] for r in conn.cursor().execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='document' "
                "AND source_id = ? ORDER BY chunk_index",
                (doc_id,),
            )
        ]
    finally:
        conn.close()


def _save(project, tmp_path, capsys, *, name, title, cite) -> int:
    body_file = tmp_path / f"{name}.md"
    body_file.write_text(f"# {title}\n\nClaim[^chunk:{cite}].", encoding="utf-8")
    save_finding.main([
        "--project", project,
        "--title", title,
        "--description", f"{title} description.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    # save_finding emits a type-tagged id ("finding:N"); return the bare int so
    # callers can build SQL params and the prefixed flag/output forms freely.
    return unprefix(out["finding_id"])


def test_merge_memory_off_foreign_source_rejected(seeded_project, tmp_path, capsys):
    """A memory-off session cannot consume a foreign session's finding, and the
    gate fires before any deletion — both findings survive."""
    project = seeded_project["project"]
    c0 = _doc_chunk_ids(project, seeded_project["doc_a"])[0]

    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        foreign_src, _ = seed_finding(conn, author)
    finally:
        conn.close()

    # A separate memory-off session owns the target but tries to fold in the
    # foreign source.
    start_session(project, memory_enabled=False)
    target = _save(project, tmp_path, capsys, name="t", title="Mine", cite=c0)

    merged_file = tmp_path / "m.md"
    merged_file.write_text(f"Body[^chunk:{c0}].", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        merge_findings.main([
            "--project", project,
            "--from", f"finding:{foreign_src}",
            "--into", f"finding:{target}",
            "--body-file", str(merged_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"
    assert out["foreign_finding_ids"] == [foreign_src]

    conn = open_db(project)
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id IN (?, ?)",
            (target, foreign_src),
        ).fetchone()[0] == 2
    finally:
        conn.close()


# ---------- skill_delete_finding ----------


def _finding_exists(project, finding_id) -> bool:
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id = ?", (finding_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_delete_finding_memory_off_other_session(seeded_project, capsys):
    """A memory-off session cannot delete a finding another session authored,
    and the foreign finding survives the rejected call."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        finding_id, _ = seed_finding(conn, author)
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    with pytest.raises(SystemExit) as exc:
        delete_finding.main([
            "--project", project, "--finding-id", f"finding:{finding_id}",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"
    # The gate fires before any delete — the finding is untouched.
    assert _finding_exists(project, finding_id)


# ---------- annotations ----------


BODY = "alpha beta. gamma beta. delta beta"


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


# ---------- annotations_skill ----------


def _run_annotations_skill(script, args, capsys) -> dict:
    script.main(args)
    return json.loads(capsys.readouterr().out)


def _run_err(script, args, capsys) -> dict:
    with pytest.raises(SystemExit) as exc:
        script.main(args)
    assert exc.value.code == 1
    return json.loads(capsys.readouterr().out)


def _annotate(project, finding_id, *extra) -> list[str]:
    return ["--project", project, "--finding-id", finding_id, *extra]


def _foreign_finding(project) -> int:
    """A finding authored by another (memory-on) session."""
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        finding_id, _ = seed_finding(conn, conn.last_insert_rowid(), body="Claim one.")
    finally:
        conn.close()
    return finding_id


def _annotation_count(project, finding_id: int) -> int:
    conn = open_db(project)
    try:
        return count_annotations(conn, [finding_id])[finding_id]
    finally:
        conn.close()


def test_annotate_memory_off_foreign_refused_own_allowed(
    seeded_project, tmp_path, capsys
):
    project = seeded_project["project"]
    foreign = _foreign_finding(project)
    start_session(project, memory_enabled=False)

    out = _run_err(annotate_finding, _annotate(
        project, f"finding:{foreign}", "--body", "x",
    ), capsys)
    assert out["code"] == "MEMORY_OFF"
    conn = open_db(project)
    try:
        assert count_annotations(conn, [foreign]) == {foreign: 0}
    finally:
        conn.close()

    own = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run_annotations_skill(annotate_finding, _annotate(project, own, "--body", "mine"), capsys)
    assert out["finding_id"] == own


def test_delete_annotation_memory_off_foreign_refused_own_allowed(
    seeded_project, tmp_path, capsys
):
    project = seeded_project["project"]
    foreign = _foreign_finding(project)
    # Annotate the foreign finding from a memory-on session first.
    foreign_note = _run_annotations_skill(annotate_finding, _annotate(
        project, f"finding:{foreign}", "--body", "theirs",
    ), capsys)["annotation_id"]

    start_session(project, memory_enabled=False)
    out = _run_err(delete_annotation, [
        "--project", project, "--annotation-id", foreign_note,
    ], capsys)
    assert out["code"] == "MEMORY_OFF"
    assert _annotation_count(project, foreign) == 1  # row survives

    own = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    own_note = _run_annotations_skill(annotate_finding, _annotate(project, own, "--body", "mine"), capsys)
    out = _run_annotations_skill(delete_annotation, [
        "--project", project, "--annotation-id", own_note["annotation_id"],
    ], capsys)
    assert out["finding_id"] == own


def test_memory_off_read_hides_foreign_agent_notes(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    start_session(project, memory_enabled=False)
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    _run_annotations_skill(annotate_finding, _annotate(project, fid, "--body", "mine"), capsys)
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES ('other', 1)"
        )
        other = conn.last_insert_rowid()
        insert_annotation(conn, finding_id=unprefix(fid), body="theirs",
                          is_human_author=False, session_id=other)
        insert_annotation(conn, finding_id=unprefix(fid), body="human",
                          is_human_author=True)
    finally:
        conn.close()

    read = _run_annotations_skill(read_finding, ["--project", project, "--finding-id", fid], capsys)
    assert [a["body"] for a in read["annotations"]] == ["mine", "human"]
    listed = _run_annotations_skill(list_findings, ["--project", project], capsys)
    [row] = listed["findings"]
    assert row["annotation_count"] == 2


def test_memory_off_delete_annotation_cannot_reach_foreign_agent_note(
    seeded_project, tmp_path, capsys,
):
    project = seeded_project["project"]
    start_session(project, memory_enabled=False)
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES ('other', 1)"
        )
        other = conn.last_insert_rowid()
        theirs = insert_annotation(conn, finding_id=unprefix(fid), body="theirs",
                                   is_human_author=False, session_id=other)
        human = insert_annotation(conn, finding_id=unprefix(fid), body="human",
                                  is_human_author=True)
    finally:
        conn.close()

    # Behind the wall a foreign agent note is indistinguishable from a missing id —
    # and it is not read back, not deleted.
    err = _run_err(delete_annotation, [
        "--project", project, "--annotation-id", f"annotation:{theirs}",
    ], capsys)
    assert err["code"] == "ANNOTATION_NOT_FOUND"
    assert _annotation_count(project, unprefix(fid)) == 2
    # A human note on the session's own finding is still deletable.
    out = _run_annotations_skill(delete_annotation, [
        "--project", project, "--annotation-id", f"annotation:{human}",
    ], capsys)
    assert out["body"] == "human"
    # And delete_finding's dropped count excludes the walled note.
    dropped = _run_annotations_skill(delete_finding, ["--project", project, "--finding-id", fid], capsys)
    assert dropped["annotations_dropped"] == 0
