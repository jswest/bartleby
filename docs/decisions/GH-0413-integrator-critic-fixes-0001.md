# GH-0413 — integrator critic-loop fixes

**Date:** 2026-06-11
**Issue:** omnibus #413 (integrator commit, no PR)

Three post-critic fixes applied directly on the omnibus branch.

## Fix 1 — narrow the runner's USAGE_ERROR arm

`#402` moved `parse_args` inside `skill_runner.run()`'s main `try` with an
`except SystemExit` arm mapping non-zero exits to a `USAGE_ERROR` envelope. That
arm caught a `SystemExit` raised *anywhere* in the try — a future `work()` (or a
library calling `sys.exit`) would be mislabeled `USAGE_ERROR`.

Now only the `parse_args(argv)` call is wrapped in its own `try/except
SystemExit`: a non-zero exit prints the `USAGE_ERROR` envelope and exits 1; a
clean exit (code 0, e.g. `--help`) re-raises so help still exits 0. Everything
past parsing (open_db, session resolution, `work`, the #407 conn-close/log_call
teardown) is unchanged. A `SystemExit` from `work()` is a `BaseException`, not an
`Exception`, so the `INTERNAL_ERROR` catch-all doesn't swallow it — it propagates
with its own code, never remapped to `USAGE_ERROR`.

## Fix 2 — pin the #411 guard roster to the dispatcher

`tests/test_skill_flag_conventions.py` hand-copies the dispatcher's script
roster. A new assertion checks `set(SCRIPTS) == set(bartleby.commands.skill.SCRIPTS)`
so a script added to the dispatcher tomorrow can't silently escape the convention
guard. Existing flag assertions unchanged.

## Fix 3 — `positive_int` on the remaining id flags

Several id flags still used bare `type=int`. Document/finding/chunk ids are
autoincrement `>= 1`, so the existing `_common.positive_int` validator is
correct — swapped in (no new helper):

- `save_date.py` `--document`, `save_summary.py` `--document`
- `delete_finding.py` / `edit_finding.py` / `read_finding.py` `--finding`
- `assign_tag.py` `--chunk`, `tag.py` `--document`, `merge_findings.py` `--into`

`--from`/`--documents` (comma-int lists) and `--limit`/`--offset` (already #403)
were left alone.

## Touched

- `bartleby/skill_runner.py`
- `tests/test_skill_flag_conventions.py`, `tests/test_skill_runner.py`,
  `tests/test_skill_save_date.py`
- `bartleby/skill_scripts/{save_date,save_summary,delete_finding,edit_finding,read_finding,assign_tag,tag,merge_findings}.py`
