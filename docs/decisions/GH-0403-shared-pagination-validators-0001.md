# Shared positive/non-negative validators for pagination + id flags (issue #403)

> Source: [#403](https://github.com/jswest/bartleby/issues/403)

`--limit`/`--offset` and id flags were hand-validated unevenly across the skill
scripts: `scan` and `search` already routed `--limit`/`--offset` through the
shared `positive_int`/`nonneg_int` argparse `type=` validators in
`bartleby/skill_scripts/_common.py`, but `read_chunks`, `list_documents`, and
`list_findings` still took bare `type=int` for their pagination flags, and
`read_document`'s `--document` id flag was bare `int`. A bare `int` accepts `0`
and negatives, which then either silently mis-paginate or surface as opaque
downstream errors.

Fix is a narrow swap, not a new abstraction: `--offset` → `nonneg_int`,
`--limit` → `positive_int`, and the `--document` id flag → `positive_int`,
reusing the existing validator pair (no new helper invented). Because every
skill funnels `parse_args` through the runner's `try`, an out-of-range value
raises `argparse.ArgumentTypeError` → `SystemExit` → the
`{"error", "code": "USAGE_ERROR"}` envelope on stdout with exit 1, *before* any
DB/session is opened or query runs (the seam established in
[#402](GH-0402-argparse-json-envelope-0001.md)). Behavior for valid inputs is
unchanged.

Deliberately **scoped down**: id-only flags on scripts that take no
`--limit`/`--offset` (e.g. `merge_findings --into`, `delete_finding --finding`,
`edit_finding`, `read_finding`, `save_summary`, `save_date`, `assign_tag`,
`tag`) were left on bare `int` — outside this issue's pagination/id-pagination
scope. No new helper, no central code module, no schema change. Each touched
script's test file gained a parametrized case asserting that an out-of-range
`--limit`/`--offset` (and a non-positive `--document`) yields the
`USAGE_ERROR` envelope with exit 1.
