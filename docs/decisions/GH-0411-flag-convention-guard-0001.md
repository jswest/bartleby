# GH-0411 — parser-introspection guard for the #111 flag conventions

**Date:** 2026-06-11
**Issue:** #411 (part of omnibus #413)

## Problem

The #111 omnibus settled a set of skill-script flag conventions, but nothing
*enforced* them structurally — they held only by reviewer vigilance. #403, #405,
and #408 each fixed a drift after the fact. Without a tripwire, the next drift
ships unnoticed.

## Decision

A new test, `tests/test_skill_flag_conventions.py`, introspects each skill
script's argparse parser and fails on a convention violation. It covers exactly
the four conventions the omnibus settled:

- **#403** — every `--limit` action is typed with the shared `positive_int`
  validator and every `--offset` with `nonneg_int` (identity check against the
  imported functions, not a bare `int`).
- **#405** — a script that identifies a *single existing tag* names that
  selector `--tag`, never `--name` / `--tag-name`. The covered scripts are
  `delete_tag`, `assign_tag`, `unassign_tag`, `extract`, `tag`. `add_tag` is
  excluded (its `--name` names a tag being *created*); `rename_tag` (`--old`/
  `--new`) and `merge_tags` (`--from`/`--into`) are excluded (they name *two*
  tags via relational flags).
- **#408** — every scope-supporting script (`search`, `scan`, `describe_corpus`,
  `list_documents`) exposes `--in-documents`.
- **#111** — no flag carries a redundant `-id` suffix; the `_id` lives on `dest`,
  not the option string (the convention is `--document` / `--finding` / `--chunk`).

## How it introspects

Each script's `parse_args(argv)` builds an `ArgumentParser` then immediately
calls `parser.parse_args(argv)`. The test patches that final call to raise with
the parser as payload, so the build runs but the consume doesn't — yielding the
parser without per-script valid argv. A script that doesn't follow this shape
(none do today) is *skipped*, not silently passed.

## Why literal, not generalized

Per the issue's anti-bloat fence, the assertions are deliberately LITERAL: they
name the exact scripts and exact flags each convention covers. The guard is a
tripwire, not a re-derivation of every script's full signature. We rejected
"extendable" generalizations (plural/arity/comma-list inference of what *should*
be a tag flag or a scope flag) because they would turn the test into a brittle
mirror of the spec — failing on benign signature changes and obscuring which
convention actually broke. When a new script joins a convention's set, add it to
the relevant literal list here; that one-line edit is the intended maintenance
cost.

## Touched

- `tests/test_skill_flag_conventions.py` — new (test-only; no production change).
