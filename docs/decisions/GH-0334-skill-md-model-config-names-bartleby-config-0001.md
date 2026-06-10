# SKILL.md names `bartleby config` for model configuration, not stale `bartleby ready`

**Issue:** #334 (part of the v0.8.11 omnibus, #331; the deferred remainder of #329's sweep)

## Context

PR #137 split the two surfaces apart: `bartleby ready` *installs/refreshes the
skill* into `~/.claude/skills/`, while the interactive provider/model wizard
moved to `bartleby config`. #329 fixed the `NO_PROVIDER` error string in
`skill_scripts/_tags.py`, but left one sibling: the agent-facing guide
`bartleby/skill/SKILL.md:134` still told the agent the summarizer model "is
configured in `bartleby ready`" — the same stale-since-#137 misdirection, in the
model-config sense.

## Decision

Change that one clause to name `bartleby config`:

```
... which model will be used (the model is configured in `bartleby config`).
```

A tree-wide audit confirmed `SKILL.md:134` was the **only** remaining
model-config-sense reference. Every other `bartleby ready` occurrence is the
*install sense* and is correct as-is — `README.md`'s install section,
`bartleby/skill/README.md`, `bartleby/commands/ready.py` (the installer command
itself), and `tests/test_ready.py` (its tests). Those were deliberately left
untouched; the dividing line is *install* (`bartleby ready`) vs *configure
provider/model* (`bartleby config`).

## Scope

Documentation-only copy fix — no code, no schema change, no re-ingest. Closes out
the sweep #329 opened; there is no third sibling to chase.
