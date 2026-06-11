# GH-0410 — dispatcher + runner JSON-contract tests

Issue #410 asked us to pin the dispatcher/runner JSON contract: UNKNOWN_SKILL,
INTERNAL_ERROR, the bare-invocation envelope (#401), the argparse-usage envelope
(#402), and conn-close/audit behavior on session-resolution failure (#407).

## What the omnibus already covered (left untouched)

By the time #410 ran, sibling issues had already landed most of the contract:

- `tests/test_skill_runner.py` (from #339/#402/#407) already pins:
  - `INTERNAL_ERROR` envelope + exit 1, with full mutation rollback across
    `findings`/`chunks`/fts/vec while the audit row survives.
  - read-only opens no transaction; `mutates=True` opens one.
  - `USAGE_ERROR` envelope on a bad flag (argparse usage dump never reaches
    stdout) — #402.
  - `--help` still exits 0 with argparse's own help — #402.
  - session-resolution failure closes the conn (no leak) and emits the envelope
    with no audit row written — #407.
- `tests/test_skill_dispatch.py` (from #401) already pins:
  - bare invocation → `MISSING_SKILL` envelope on stdout, prose to stderr, exit 1.
  - `-h/--help` → exit 0 with help text.
  - unknown name → `UNKNOWN_SKILL` envelope + exit 1.
  - `SCRIPTS` tuple is non-empty.

## What #410 added (the genuine gaps)

- **Runner — `NO_ACTIVE_PROJECT` envelope path.** No `--project` and no active
  project (the autouse home-isolation fixture guarantees a fresh sandbox) now
  asserts the `NO_ACTIVE_PROJECT` envelope + exit 1, *and* that `open_db` is
  never called — the SkillError fires before the DB is opened, so there is no
  conn to leak and no audit row. This was the one runner error arm with no test.
- **Dispatcher — routing + remaining-argv passthrough.** A known name routes to
  `bartleby.skill_scripts.<name>.main` with the name consumed and the rest of
  argv handed through verbatim.
- **Dispatcher — exit-code propagation.** A script that `sys.exit(2)` surfaces
  code 2 to the caller unchanged; the dispatcher does not swallow `SystemExit`.

## Explicitly NOT done (hard fence)

Did not build the fragile parametrized "induce a failure in all ~23 scripts"
sweep — per-script envelopes are pinned by their own tests. Smallest fix that
fits: four new cases, no renames, no duplication.

## Gate

`uv run pytest tests/test_skill_runner.py tests/test_skill_dispatch.py` → 14
passed; full `uv run pytest` → 867 passed.
