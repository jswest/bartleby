# Exempt `[YYYY]`-shaped neutral case citations from the malformed-citation guard (issue #698)

> Source: [#698](https://github.com/jswest/bartleby/issues/698)

## Problem

`reject_malformed_citations` in `bartleby/skill_scripts/_common.py` treats any
`\[\^?(\d+)\]` as a citation-shaped marker missing its `[^chunk:N]` / `[^finding:N]`
type tag, and rejects the whole save with `MALFORMED_CITATION`. Common-law
neutral case citations are bracketed-year by construction (`[1998] HKLRD 771`),
so ordinary prose quoting case law false-trips the guard. The escaping
workaround (`\[1998\]`) has a second-order failure: prose that explains the
escape by quoting the bare form trips the same check again on the next save.

## Fix

Exempt the caret-less, exactly-4-digit bracket shape — `[1998]` — from the
guard. That shape is never a valid chunk-id marker in practice (chunk ids
climb well past 4 digits in any real corpus, and the untyped-marker convention
this guard polices is `[N]` for an arbitrary-length id, not specifically
4-digit ids). Implemented as a second regex, `_BRACKETED_YEAR = r"^\[\d{4}\]$"`,
checked against each `_MALFORMED_MARKER` match before it's added to the `bad`
list.

## Design choice: the exemption is caret-less only

The issue's repro and suggested fix only mention the bare `[1998]` form. The
existing guard also catches `[^N]` — described in its own comment as "the
now-obsolete bare chunk form" (pre-#624 syntax, before citations were required
to carry a `[^chunk:N]` type tag). The question: does a 4-digit ref *with* a
caret — `[^1998]` — get the same exemption?

**Decision: no.** The exemption stays caret-less-only; `[^1998]` is still
rejected as `MALFORMED_CITATION`.

Reasoning:

- Neutral case citations never carry a caret — `[1998] HKLRD 771` is the
  entire real-world shape this issue is about. There's no case-law citation
  convention that would produce `[^1998]`, so extending the exemption there
  buys no prose-compatibility win.
- `[^N]` is a *specific* legacy pattern (the pre-#624 bare-chunk-citation
  syntax) that the guard exists to catch precisely because it's easy to type
  by accident (muscle memory from the old convention) and would otherwise
  silently drop the citation — the exact failure #624 was opened to kill.
  Widening the exemption to `[^1998]` would silently swallow a real instance
  of that failure whenever the stale chunk id happened to be 4 digits, which
  is not a rare coincidence in a corpus with a few thousand chunks.
- Asymmetric exemption (caret-less only) is the minimal-surprise reading of
  the issue's own suggested fix, which names the bracket shape without a
  caret, and it costs nothing in practice since no real citation format
  produces the careted form.

This is implemented as a side effect of the regex, not a separate branch:
`_BRACKETED_YEAR` requires `\[` immediately followed by `\d{4}` and `\]`, so
`_MALFORMED_MARKER`'s `[^1998]` match (whose `group(0)` is `"[^1998]"`, caret
included) never matches `_BRACKETED_YEAR` — no extra caret-detection logic
needed.

## Regression coverage

`tests/test_external_citations.py`:
- `test_malformed_check_exempts_bracketed_year_neutral_citation` — bare
  `[1998]` in prose alongside a valid `[^chunk:N]` citation is accepted.
- `test_malformed_check_still_rejects_non_four_digit_bracket` — `[12]`
  (non-4-digit) is still rejected.
- `test_malformed_check_still_rejects_careted_four_digit_bracket` — `[^1998]`
  (4-digit, with caret) is still rejected, pinning the asymmetric-exemption
  decision above.
