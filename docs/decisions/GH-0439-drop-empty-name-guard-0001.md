# Drop the `EMPTY_NAME` guard in `add_tag`/`rename_tag` (issue #439)

> Source: [#439](https://github.com/jswest/bartleby/issues/439)

Both `add_tag` and `rename_tag` ran two sequential guards on the same input:
`if not name: raise EMPTY_NAME` immediately followed by
`if not normalize_name(name): raise EMPTY_NORMALIZED_NAME`. The name is already
`.strip()`-ed before either guard, and `normalize_name` deletes every
non-`[a-z0-9]` character (`_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")`), so
an empty or whitespace-only name normalizes to `""` (falsy) and is *always*
caught by the second guard. The `EMPTY_NAME` branch therefore guarded a state
the `normalize_name` check already covered — there is no input that fires
`EMPTY_NAME` where `EMPTY_NORMALIZED_NAME` would not. Dead code by definition.

So the `EMPTY_NAME` branch is removed from both scripts, leaving the single
`normalize_name` guard. A blank or whitespace-only tag name now reports
`EMPTY_NORMALIZED_NAME` ("must contain at least one alphanumeric character"),
which is a truthful description of the blank case too. That is fine because
nothing in the repo — tests, `SKILL.md`, or other scripts — distinguished the
two codes (repo-wide grep confirms no surviving reference to `EMPTY_NAME`), and
the agent's remediation is identical either way: supply a real name. The only
loss is a marginally more specific message for the all-whitespace sub-case.

`add_tag`'s separate `EMPTY_DESCRIPTION` guard is unrelated and stays. Tests now
pin the surviving envelope: whitespace-only and punctuation-only names fail with
the `EMPTY_NORMALIZED_NAME` JSON envelope on both paths. Pure deletion, no schema
change.
