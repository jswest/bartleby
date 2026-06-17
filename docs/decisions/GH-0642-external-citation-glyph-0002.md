# External citations get the § glyph, distinct from the chunk dagger (issue #642)

> Source: [#642](https://github.com/jswest/bartleby/issues/642)

The finding-detail citation gutter used **†** for both corpus-chunk citations
*and* external `[^url:…]`/`[^doc:…]` citations, so the reader had no way to tell
"this points at a chunk in your corpus" from "this points at an outside document
or URL." Only the gone state was distinct (**‡**).

External citations now render **§**. The three glyphs follow the traditional
footnote-mark sequence `* † ‡ §`: **† corpus chunk**, **‡ chunk whose source is
gone**, **§ external (url/doc)**. § ("section") reads as a document/reference mark
and is visually heavier than a superscript asterisk, the other candidate.

`url` and `doc` share the one § glyph rather than splitting into two marks —
three legible concepts beat four, and the gutter note already distinguishes them
(a URL renders as a link, a doc as a muted ref). The change is the glyph only:
the `cite-ref--source` tone, the `data-note` ordinal tie-back, and the click
behaviour are unchanged. Web-only, no `SCHEMA_VERSION` bump.
