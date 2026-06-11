# GH-0438 — delete `validate_value_type`; argparse `choices=` already guards it

**Date:** 2026-06-11
**Issue:** #438 (part of omnibus #447)

## Decision

`validate_value_type` in `_tags.py` is deleted, and the `INVALID_VALUE_TYPE`
error code is removed from the vocabulary. `add_tag.work()` now reads
`args.value_type` directly.

## Why the guard is unreachable

`validate_value_type`'s only caller in the entire repo was `add_tag.work()`. Its
job was to raise `INVALID_VALUE_TYPE` when `args.value_type` was anything outside
`VALUE_TYPES = ("number", "string", "date")`. But `add_tag`'s argparse
declaration already pins that flag:

```python
p.add_argument(
    "--value-type", dest="value_type", choices=list(VALUE_TYPES), default=None, ...
)
```

`choices=list(VALUE_TYPES)` makes argparse reject any other value during
`parse_args` — and `parse_args` runs *outside* (before) the runner's `work()`
call (`skill_runner.run` calls `parse_args` at the top, then `work` only after;
see `GH-0402-argparse-json-envelope-0001.md`). So by the time
`validate_value_type(args.value_type)` would have run, `args.value_type` is
provably `None` or a member of `VALUE_TYPES`. The `raw not in VALUE_TYPES` branch
can never fire — the function was a dead identity check and `INVALID_VALUE_TYPE`
was an unreachable code.

## What the user sees (unchanged behavior surface)

An invalid `--value-type` no longer produces a JSON `{"code": "INVALID_VALUE_TYPE"}`
envelope; argparse's `choices=` rejects it before `work()` runs, exiting non-zero
with a `USAGE_ERROR` envelope (per GH-0402) whose stderr usage message names the
allowed values (number/string/date). That is fine because it is already the error
surface for every other malformed flag in the skill scripts — missing required
flags, non-integer ids, and enum flags like `list_documents --order` and
`scan --group-by` all surface as argparse usage errors. No test or SKILL.md text
referenced `INVALID_VALUE_TYPE`, and the agent-facing outcome (rejection with a
message listing the valid types) is preserved.

## No back-compat

Per the repo's no-backwards-compatibility default, `validate_value_type` and its
import are deleted outright — no shim.

## Touched

- `bartleby/skill_scripts/_tags.py` — `validate_value_type` removed (13 lines).
- `bartleby/skill_scripts/add_tag.py` — import dropped; `work()` reads
  `args.value_type` directly.

Net: shrink. Full suite: green.
