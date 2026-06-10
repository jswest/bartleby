# `NO_PROVIDER` error names `bartleby config`, not stale `bartleby ready`

**Issue:** #329 (part of the v0.8.11 omnibus, #331)

## Context

`resolve_classifier()` in `bartleby/skill_scripts/_tags.py` raises a
`SkillError("NO_PROVIDER", ...)` when the summarizer config has no provider or
model. Its message told the user to run `bartleby ready` — but PR #137 split the
two surfaces apart: `bartleby ready` now only *installs the skill* into
`~/.claude/skills/`, while the provider-config wizard moved to `bartleby config`.
So the remedy the error named no longer fixes the error.

## Decision

Change the one message string to point at the correct command:

```
"No LLM provider configured. Run `bartleby config` first.",
```

A focused test (`test_resolve_classifier_no_provider_names_config_command` in
`tests/test_skill_tags.py`) monkeypatches `load_config` to an empty dict and
asserts the raised `SkillError` has code `NO_PROVIDER`, contains `bartleby
config`, and does *not* contain `bartleby ready` — so the fix can't silently
regress to the stale command.

## Scope

Deliberately just this one message. Auditing other stale `bartleby ready`
references elsewhere in the tree is a separate sweep item and was left untouched
here.
