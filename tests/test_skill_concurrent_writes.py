"""Concurrent finding-write regression test for the runner's mutating seam (#562).

Several agents (or one agent issuing rapid calls) can drive ``save_finding`` /
``edit_finding`` into the *same* session at once. Each invocation is a separate
process/connection, but they all contend for the one project DB's single write
lock.

The bug (#562): the mutating seam in ``skill_runner.run`` used to open the
transaction with apsw's ``with conn:`` — a *deferred* transaction that takes no
write lock until its first write. A ``work`` that reads before it writes (every
finding write does: it resolves the session, validates citations, …) then hits
SQLite's read-to-write upgrade trap, where the upgrade returns ``SQLITE_BUSY``
*immediately*, ignoring ``busy_timeout``. So a handful of concurrent writers
produced spurious first-attempt ``BusyError``\\ s instead of serializing.

The fix opens the seam with ``BEGIN IMMEDIATE``, taking the write lock up front
so ``busy_timeout`` governs the wait and concurrent writers block-and-serialize.
This test spawns N concurrent writers against one sandbox project DB and asserts
none fail (in particular, none fail with a ``BusyError``). It is verifiable
in-suite: no live corpus, no network.
"""

from __future__ import annotations

import json
import sys
import threading

from bartleby.db.connection import open_db
from bartleby.skill_scripts import edit_finding, save_finding
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
)


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


def test_concurrent_save_findings_serialize(seeded_project, tmp_path):
    """N threads saving findings into one session DB all succeed (serialize)."""
    project = seeded_project["project"]
    cited = _cited_chunk(project, seeded_project["doc_a"])

    n = 6
    argvs = []
    for i in range(n):
        bf = tmp_path / f"save_{i}.md"
        bf.write_text(f"Concurrent claim {i}[^{cited}].", encoding="utf-8")
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
    return json.loads(buf.getvalue())["finding_id"]


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
        bf.write_text(f"Seed body {i}[^{cited}].", encoding="utf-8")
        finding_ids.append(_save_one(project, f"edit-seed-{i}", bf))

    argvs = []
    for i in range(n):
        bf = tmp_path / f"edit_{i}.md"
        bf.write_text(f"Edited body {i}[^{cited}].", encoding="utf-8")
        argvs.append([
            "--project", project,
            "--finding-id", str(finding_ids[i]),
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
            assert body == f"Edited body {i}[^{cited}]."
    finally:
        conn.close()
