# GH-0405 — align `delete_tag` flag with its tag-taking siblings

**Date:** 2026-06-11
**Issue:** #405 (part of omnibus #413)

## Decision

`delete_tag`'s tag-identifying flag is renamed `--name` → `--tag`, matching the
convention its siblings already use to name an **existing** tag:

- `assign_tag --tag <name>`
- `unassign_tag --tag <name>`

`delete_tag` was the odd one out, taking `--name` for the existing tag it drops.
`--name` is reserved across the tag scripts for the tag being *created*
(`add_tag --name <n> --description <d>`), so reusing it to identify an existing
tag was a genuine inconsistency, not a defensible distinction.

`rename_tag` (`--old`/`--new`) and `merge_tags` (`--from`/`--into`) name two tags
each and keep their relational flags — `--tag` is the canonical name only where a
single existing tag is identified, which is exactly `delete_tag`'s shape.

## No back-compat

Per the repo's no-backwards-compatibility default, the old `--name` flag is
deleted outright — no alias, no deprecation. Breaking the skill flag name is
acceptable here.

## Touched

- `bartleby/skill_scripts/delete_tag.py` — `--name` → `--tag` (arg + `args.tag`).
- `bartleby/skill/SKILL.md` — usage row updated to `delete_tag --tag <name>`.
- `tests/test_skill_tags.py` — `test_delete_tag_cascades_assignments` invocation.

Full suite: 833 passed.
