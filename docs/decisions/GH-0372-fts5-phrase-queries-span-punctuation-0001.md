# SKILL.md documents that FTS5 phrase queries span punctuation (issue #372)

> Source: [#372](https://github.com/jswest/bartleby/issues/372)

FTS5 tokenizes on punctuation, so a quoted phrase matches a token *sequence* and reaches *across* intervening punctuation: `"foo bar"` matches `foo bar`, `foo, bar`, and `foo. Bar` alike. This is load-bearing for precise extraction on templated corpora — it's how an agent turns a semi-structured annotation (e.g. `Smith (Director of Operations)`) into an exact phrase filter by quoting just its words — but it was documented nowhere, so each agent rediscovered it by luck or never at all. Docs-only: one short paragraph teaching the technique plus its inverse caveat. No code, no schema change, no `SCHEMA_VERSION` bump.

## Where it lives: the scan row, not "Reading search results"

The behavior is a property of how you *construct* a query, so it belongs with the tool whose job is literal-phrase matching. `scan` is FTS5-only and its row already discusses phrase vs. `--match-terms` token matching, making it the natural anchor; "Reading search results" is about triaging the result set (`rank`/`normalized_score`/`authored_date`), a different concern. The paragraph is appended to the existing `scan` row in the Available-scripts table, row-scoped, touching no other section. (The same FTS5 phrase semantics also govern `search`'s keyword leg, but the scan row is where templated-phrase pinning is taught, so one mention there is enough — no need to repeat it under `search`.)

## The inverse caveat is the half that prevents the wrong mental model

Stating only "phrases span punctuation" invites the dual error: trying to *require* a bracket or comma. Punctuation is not a token, so it can never appear inside a phrase or be required by a query. The paragraph states both halves so an agent doesn't try to anchor on `(` after learning the phrase reaches through it.

## README mirrors it, row-scoped

`bartleby/skill/README.md` carries a parallel scan row ("matching a literal phrase corpus-wide"), so it gets the same note in condensed form to keep the two surfaces from diverging. Kept to one sentence inside the existing row — the README is an overview, not the agent-facing contract, so it doesn't carry the full worked example.

## Generic example, not corpus-specific

The example uses `foo, bar` / `foo. Bar` and a generic `Smith (Director of Operations)`, deliberately not any real corpus's marker phrase — the behavior is a property of FTS5, not of any corpus, and SKILL.md examples stay corpus-agnostic.
