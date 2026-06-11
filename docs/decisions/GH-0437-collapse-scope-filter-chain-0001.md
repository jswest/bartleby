# GH-0437 — collapse the single-caller scope-filter indirection chain in _tags.py

Issue #437 (tier-2 dry-sweep judgment bet) finished the residue left after #432
inlined `intersect_tag_filter` / `intersect_file_like_filter` into
`resolve_scope`: it folds `Scope.filters_dict` into its sole caller
`Scope.echo_into` and corrects two docstrings that still advertised consumers
which no longer exist.

## What changed

- `Scope.filters_dict` (the 18-line "build the self-describing `filters` echo,
  or `None` when unfiltered" method) is gone. Its dict literal now lives inline in
  `Scope.echo_into`, guarded directly by `if self.active:` — `echo_into` was its
  only caller (whole-repo grep), and every script that surfaces a filters echo
  (`search`, `scan`, `describe_corpus`, `list_documents`) calls `echo_into`, never
  `filters_dict`.
- `resolve_tag_names`'s docstring no longer claims "Used by `list_documents`,
  `search`, and any future consumer of `--tag`." Its only caller is
  `resolve_scope` (those scripts route their `--tag` exclusively through
  `resolve_scope`); the docstring now names it the query unit behind
  `resolve_scope`'s `--tag` intersection.
- The `Scope` class docstring's "self-describing via `filters_dict`" now points at
  `echo_into`, the surviving entry point.

No call sites changed: the four consumers already called `echo_into`. The emitted
`filters` dict is byte-for-byte identical — same keys, same order, same
`if self.active` gate — so scope echo semantics are unchanged.

## Why collapsing is correct

`intersect_tag_filter`, `intersect_file_like_filter`, and `filters_dict` were
built as reusable seams: the two intersect helpers so scripts could compose
tag/file-like filtering individually, `filters_dict` as a shared echo formatter.
But `resolve_scope` became the single documented entry point for scoping, and
every current consumer (`search.py`, `scan.py`, `list_documents.py`,
`describe_corpus.py`) routes through it; the only `filters_dict` caller is
`echo_into` directly below it. The seams each had exactly one caller and
docstrings naming consumers that had migrated away — speculative generality.

After this collapse there is no standalone tag-intersection, file-like-intersection,
or filters-echo helper. A future script wanting one of those pieces *without* full
scope resolution would have to call `resolve_scope` or re-extract a few lines. That
is the right trade: all four current consumers already use `resolve_scope` as the
single entry point, the underlying named query units (`documents_with_any_tag`,
`documents_matching_file_like`, `resolve_tag_names`) remain available, and this
repo's smallest-fix culture extracts the seam when the second caller actually
arrives rather than carrying it speculatively.

## What was NOT touched

The memory wall is untouched. `memory_enabled=0` excluding prior-session findings
is enforced in `search.py` at script level (load-bearing per CLAUDE.md); this change
is confined to `_tags.py` scope-echo plumbing and does not reach that path. The
`document_ids` resolution (the actual filter set, tags ∩ in_documents ∩ file_like ∩
date bounds) is unchanged — only the echo-formatting indirection collapsed.

---
*Filed from the 2026-06-11 dry sweep (dead/wet/bloat audit; every item adversarially
verified by an independent defender pass).*
