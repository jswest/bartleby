# The cited passage is baked into the sandboxed `srcdoc` as a `<mark>`, and the pane opens there via CSS `scroll-initial-target`. No script, no CSP change.

> Source: [#754](https://github.com/jswest/bartleby/issues/754)

On `/findings/[id]`, clicking a citation to a `.txt`/`.md` source loaded the
source pane at the top of the document, so the reader had to hunt for the
passage. The pane is `<iframe srcdoc sandbox="">` with a
`default-src 'none'; style-src 'unsafe-inline'` CSP (GH-0121, GH-0567,
GH-0680). That gives it an opaque origin with no scripts, so the parent can't
reach in and scroll it after load. The constraint agreed with john was to keep
all of that as it is: no `allow-scripts` and no CSP relaxation. So the
highlight and the scroll target both have to be in the `srcdoc` when it's built.

**Locating the passage.** `chunks` stores no source offsets, and adding them
would mean a schema bump and a re-ingest. So the chunk's text (now carried on
each citation by `getCitations`) is located in the rendered source. Chunk text
isn't byte-identical to the source, in three ways:

- `.txt` ingest folds CRLF, and its chunker cuts at character offsets, so the
  first and last words can be fragments.
- Docling markdown chunks keep inline syntax (`**x**`, `[text](url)`), but
  list bullets and line breaks come out differently.
- Docling prefixes each chunk with its heading path, one line per heading,
  and those headings aren't contiguous in the source.

So matching runs **rendered text against rendered text**. The passage goes
through the same `render` as the source: `marked.parse` for markdown, an
escaped `<pre>` for text. Both sides are read with `textNodes(root,
{ forMatch: true })`, which puts a `\n` between text nodes split by a `<br>`
or a block boundary (otherwise `one<br>two` reads as "onetwo"), and skips text
inside raw-text elements (`<style>`, `<textarea>`, `<title>`, `<script>`,
`<noscript>`, …), where a `<mark>` would re-parse as literal text.

`locatePassage` then compares them as word sequences (`[\p{L}\p{N}]+`),
ignoring whitespace, punctuation and case. It's one case-insensitive regex
built from the passage's words (`w1[^\p{L}\p{N}]+w2…`, flags `iu`), run over
the source text. That has two consequences. Nothing is materialised per
source word. On a 20 MB source the match takes a few milliseconds, where the
old word array took about 0.4 s and hundreds of MB of heap on each click. The
whole highlight, including parsing and serialising the body, takes about 70 ms
in Chromium, so there's no size cap. The cost is source length × passage
length when a passage nearly matches everywhere and then misses. A
repetitive table chunk that misses its zero-filled source can block for about a
second, which is accepted. A regex too long for V8 to compile returns no
highlight. And the offsets are the
source's own: lowercasing first would shift them, because `İ` lowercases to
two code units. The regex isn't anchored at its ends, so a fragment first word
matches the tail of a word and a fragment last word matches its head. If the
whole chunk doesn't match, leading lines are dropped one at a time, at most
six (one per heading level), to peel off the heading path. The first hit wins.

**Every attempt needs at least 5 words and at least half the chunk's words.**
Without that floor, the fallback dropped lines until even a one-word remainder
matched. A chunk `H1\nH2\nH3\n5` highlighted a stray "5", and a fragment
`he` matched inside "The". A short needle recurs by chance and the first
occurrence wins, so a short match is likely the wrong place. Five words, with
the end words possibly fragments, leaves at least three exact interior words.
The half-the-chunk rule keeps dropping limited to what it's for, peeling a
short heading path off a long body. If peeling would leave less than half the
chunk, the remainder isn't the cited passage any more. Failing either rule
means no highlight, and the pane opens at the top as before. A missed
highlight is far better than a confidently wrong one. A chunk whose text
docling reorders (table serializations, for example) won't match, and that's
accepted.

**Inserting the mark.** For markdown the mark has to survive rendering, so it
goes in *after* rendering. `markPassage` (in `viewerDoc.js`) parses the rendered
body into a `<template>`, walks its text nodes with `textNodes`, and wraps the
matched span in `<mark class="cite">`. The span can cross elements, so there
is one mark per text node it touches. Whitespace-only nodes are skipped, because a `<mark>`
between table rows would be foster-parented out of the table. The `.txt`
literal-escape path from GH-0680 is unchanged: the mark wraps text nodes of
the already-escaped `<pre>`, so nothing is ever interpreted. The finding page
renders the source client-side, so this needs only the browser's DOM.
`wrapViewerDocument`'s signature is unchanged, and the HTML export doesn't
highlight.

`textNodes` is shared with the finding page's annotation highlighting (#734),
which uses it in its default plain mode: no separators and no filter. That
mode's `text` is exactly the body's `textContent`, which is what
`Range.toString()` and the annotation round-trip's offsets are measured
against. Separators or skipped nodes there would make the live text differ
from the round-trip text, and every annotation highlight would silently drop.

**Scrolling without a script: `scroll-initial-target: nearest`** (CSS Scroll
Snap 2, Chromium 133+). It's set on a zero-size `inline-block` `.cite-anchor`
prepended to the first mark, and `html { scroll-padding-top: 3rem }` leaves a
little context above the passage. Verified in Chromium 158 headless against a
sandboxed `srcdoc` iframe under the GH-0567 CSP. Two details were load-bearing:

- The target must be an atomic inline box. On a plain inline `<mark>`,
  Chromium scrolled about 13px and stopped.
- It scrolls only the *nearest* scroll container. The viewer CSS's
  `pre { overflow-x: auto }` made every `<pre>` a scroll container, so `.txt`
  sources didn't move at all. That rule was redundant, because GH-0680's
  `pre-wrap` and `overflow-wrap: anywhere` already keep lines inside the pane,
  so it was dropped.

Browsers without the property (Firefox, Safari as of writing) still show the
highlight, but open at the top.

Rejected:

- **CSS scroll-snap.** `proximity` doesn't fire from position 0. `mandatory`
  with a single snap point pins the reader to it. Giving every block a snap
  point snaps to the first one.
- **Fragment navigation** (`<meta http-equiv=refresh>` to `#cite`, or an
  `autofocus` element). The first is a navigation hack inside the sandbox. The
  second is blocked by `sandbox=""`'s automatic-features flag.
- **`allow-scripts` or a CSP relaxation.** Excluded by the constraint.

Frontend-only; no `SCHEMA_VERSION` bump (issue #754).
