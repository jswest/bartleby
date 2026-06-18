# `scan --count-by 'file_name:/regex/'` — per-document field histogram

Issue #641 (part of omnibus #656).

## Problem

Corpora with structured filenames (member prefixes, date stamps, doc-type codes)
are common in EDGAR/OGE/PACER-style datasets. Getting a "documents per
filename-prefix" histogram previously meant reaching for `list_documents
--returning file_name` piped into a shell `grep | sort | uniq -c` loop — a
pattern brittle enough (whitespace, encoding, OS portability) to justify a
primitive.

## Decision

Extend `--count-by` to accept a `field_name:/regex/` form — a colon-separated
prefix that tells scan to apply the capture regex to a **document column** rather
than chunk body text. Initial support covers `file_name` only; no generic column
engine is built speculatively.

### Syntax choice

`--count-by 'file_name:/^([A-Za-z]+)_/'` (field-prefixed) was chosen over a
separate `--count-over <field>` selector. All histogram modes stay in one flag;
the field prefix is unambiguous and self-documenting; and the existing `document`
keyword is untouched.

### Per-document, not per-match

The body-text `--count-by '/regex/'` mode counts **per match** (a chunk with two
matches contributes two) — an honest frequency for the "how often does this term
appear" question. For a filename histogram the question is "how many documents
match each prefix" — so the field mode counts **per document**: one capture per
document, ignoring multiple regex matches within the same filename. Double-counting
would give misleading totals with no way to opt out.

### FTS scope intentionally not applied

The field histogram runs against `SELECT file_name FROM documents` (all documents
in the project DB), not against the FTS5 match set. The FTS query is required by
`scan`'s interface but the filename histogram is query-independent — the user is
asking "what's in my corpus" not "which FTS-matching documents have this prefix."
Applying the FTS scope would silently restrict the histogram to documents that
happen to contain the FTS query term, producing a subtly confusing partial
histogram. This is the deliberate semantics.

### Output envelope

A distinct output envelope (`count_by_field` / `count_by_regex` / `buckets`)
rather than reusing the body-text envelope (`count_by` / `groups`) avoids
ambiguity: a caller can always tell which mode produced the response. The bucket
shape (`{value, count}`) and rollup fields (`distinct_value_count`,
`total_match_count`, `truncated`, paginated by `--limit`/`--offset`) mirror the
body-text mode for consistency.

## Implementation

- `_common.py`: `parse_field_capture(value, *, flag)` — parses
  `<word>:/regex/` form, returns `(field_name, CaptureSpec)` or `None` if not
  field-prefixed. Validates the field name against `_COUNT_BY_FIELD_NAMES`
  (currently `("file_name",)`) before parsing the regex.
- `scan.py`: `_count_by_field(cur, spec)` — walks `documents`, applies
  `spec.pattern.search(file_name)`, takes the first capture group, accumulates a
  `Counter`. Honors the existing `CAPTURE_MAX_MATCHES` cap and returns
  `(counter, truncated)` matching `_count_by_regex`'s signature.
- `_parse_count_by` extended to a 3-tuple `(mode, spec, field)` — `field` is
  `None` for the `document` and `regex` modes.

## Scope

Non-schema, additive. Touches `bartleby/skill_scripts/_common.py`,
`bartleby/skill_scripts/scan.py`, and `tests/test_skill_scan.py`.
