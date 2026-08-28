# The project-name echo lives in `skill_runner.run()`, not `_common.py`'s `session_provenance` — and the resolved name reaches `work()` by mutating `args.project`, not by widening every script's signature.

Issue #696 (robbarry's live incident, and the earlier repro in the issue
body): `active_project` lives in the shared global `~/.bartleby/config.yaml`,
so another process on a multi-agent machine can repoint it mid-session. When
that happens, `read_chunks` reports every requested id as `missing` and
exits 0 — no warning, nothing naming the resolved project — and a
wrong-project `save_finding` can silently succeed against unrelated text.
The issue asked for two behavior-preserving fixes: warn on a 100%-miss read,
and have every skill script's output echo the resolved project. Two
placement calls were made getting there.

**The envelope addition goes in `skill_runner.run()`, not the
`session_name`/`model`/`harness` helper in `_common.py`.** The issue's own
text points at `_common.py`'s session-metadata helper (`session_provenance`)
as the precedent, and it's used by `save_finding`/`edit_finding`/
`read_finding`/`merge_findings` — but only those four, each opting in to
per-finding authorship attribution. It is not a place that touches every
skill script. `skill_runner.run()`, on the other hand, is the one function
every skill script's `main()` already calls, and it already stamps a `"run"`
key onto every successful result (the `run_key`/`model`/`harness` echo added
for #547). That's the actual "common output envelope" for the ~two-dozen
scripts, so `"project"` was added there, next to `"run"`, in the same
`if error_envelope is None and isinstance(result, dict)...` branch — one
line, uniform across every script, no per-script edits.

**`read_chunks`'s 100%-miss warning reads the resolved project off
`args.project`, which `run()` now overwrites with the resolved name — rather
than widening `work()`'s call signature to take a `project` kwarg.** Before
this change, `args.project` was `None` on the overwhelmingly common path (no
`--project` flag, riding the active-project pointer) — exactly the case the
incident describes, since nobody explicitly points at the wrong project;
they just don't override an already-wrong active pointer. Passing `project`
into `work(conn=conn, args=args, session_id=session_id)` as a new kwarg
would break every other script's `work()` signature (`def work(*, conn,
args, session_id)`, none accepting `**kwargs`) with a `TypeError`, forcing a
signature edit across ~20 files for one caller that needs it. `args` is
already threaded into every `work()`; mutating `args.project = project`
right after resolution (before `open_db`) makes the resolved name available
wherever `args` already reaches, at the cost of one attribute assignment.
Nothing reads `args.project` today expecting the pre-resolution `None`,
confirmed by grepping every `work()` in `bartleby/skill_scripts/` before
making the change.

**Scope confirmed narrow: only `read_chunks`'s `--chunks` (direct id lookup)
mode gets the warning.** Its other two modes (`--document-id`,
`--around-chunk`) already fail loudly on an unresolved id (`DOCUMENT_NOT_FOUND`
/ `CHUNK_NOT_FOUND`), so they don't have the issue's silent-`missing`-with-
exit-0 shape. `extract.py` also computes a `missing` list from a batch of
chunk ids, but it's a mutating script whose richer per-id diagnostics
(`no_match`/`cast_errors`/`conflicts`) already make an empty `stored` list
visible, and the issue's repro and fix proposals are both `read_chunks`-
specific — extending to `extract` was left out as broadening scope beyond
what was asked.
