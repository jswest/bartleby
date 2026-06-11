# `rename_tag` applies `add_tag`'s normalized-name dup check (issue #404)

> Source: [#404](https://github.com/jswest/bartleby/issues/404)

`add_tag` guards against duplicates through `find_similar_tag`, whose
normalized-name leg collapses `"NYSEG"`, `"nyseg"`, and `"ny-seg"` to one key
and refuses the second. `rename_tag` had only an *exact*-match lookup
(`get_tag_by_name`), so renaming `"ny-seg"` → `"NYSEG"` onto an existing
`"NYSEG"` slipped past and created a normalized-duplicate pair — exactly what
`add_tag` exists to prevent. This aligns the two paths: `rename_tag` now runs
the same normalized-equality check via a new `_tags.find_tag_by_normalized_name`
helper (built on the existing `normalize_name` + `fetch_vocabulary`), and raises
the same `TAG_EXISTS` envelope on collision. The `tag_id != target` guard is
retained so a case/punctuation-only *self*-rename of the same tag (e.g.
`"ny-seg"` → `"NYSEG"`) is still allowed. Deliberately scoped to the
normalized-equality leg only — no embedding/similarity check on rename (renaming
is a deliberate act on a named tag, not a fuzzy proposal like `add_tag`), and no
new abstraction beyond the one shared helper. Additive, no schema change;
`merge_tags` remains the path to actually combine two distinct tags.
