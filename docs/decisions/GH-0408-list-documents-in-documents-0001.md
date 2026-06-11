# `list_documents` — add `--in-documents` scope flag (issue #408)

> Source: [#408](https://github.com/jswest/bartleby/issues/408)

`list_documents` was the lone scoping script missing the `--in-documents`
convention its siblings (`search`, `scan`, `describe_corpus`) all carry per
#111. The plumbing was already in place: `resolve_scope` in `_tags.py` accepts
`in_documents` and the resulting `Scope` already echoes it into the `filters`
object — `list_documents` simply never defined the flag nor passed the value
through. Fix is the smallest that fits: copy `scan`'s arg block verbatim (the
`comma_int_list("document_id")` `--in-documents` flag, `dest=in_documents`) and
pass `in_documents=args.in_documents` into the existing `resolve_scope` call.
No new helpers, no schema touch — purely additive. Default (unscoped) behavior
is unchanged: without the flag `args.in_documents` is `None`, so `resolve_scope`
sees no scope and the `filters` echo stays absent. SKILL.md's `list_documents`
row now names `--in-documents` alongside `--tag` / `--file-like`.
