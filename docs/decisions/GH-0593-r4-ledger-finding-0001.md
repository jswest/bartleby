# Margin-note citations: a stacked gutter, not per-marker float (issue #593)

> Source: [#593](https://github.com/jswest/bartleby/issues/593)

R4 gives `/findings/[id]` a "ledger" treatment: a Doto drop-cap, Iosevka
hed/dek, a ~62–66ch prose measure, and **margin-note citations** — the inline
citation chips left the prose and became a right-hand gutter of sidenotes, each
tied to an inline dagger anchor at the cited point.

**The layout decision: a two-column CSS grid with the notes stacked in reading
order, NOT Tufte-style per-marker vertical alignment.** True margin notes
(each note floated to sit beside the exact line of its marker) need
JS-measured absolute positioning that fights `position: sticky`, reflows on
every resize/font-load, and collapses incoherently when two markers land on
adjacent lines. Instead the finding body and a `.cite-notes` aside are the two
tracks of a `.ledger-column` grid (`minmax(0,36rem)` prose + `minmax(0,14rem)`
gutter); notes stack in document order in the gutter. **The tie-back is carried
by the markup, not by geometry**: each inline anchor is
`<sup class="cite-ref cite-ref--source|--gone" data-note="N">†N</sup>` and its
note is `<div class="margin-note margin-note--source|--gone" data-note="N">`
with a matching `†N`/`‡N` dagger badge. The dagger glyph + ordinal is the
visual pair; `data-note` is the programmatic pair (a click on the anchor
`scrollIntoView`s the note and activates its source in the viewer). This is
robust, reflow-free, and degrades cleanly.

**Graceful narrow-screen degradation falls straight out of the grid.** At
`max-width: 56rem` the `.ledger-column` collapses to one column, so the gutter
folds *below* the prose as a stacked end-note list (a top rule replaces the
left rule). The daggers still tie inline marker → note, so no information is
lost on mobile — the citations just become footnotes instead of sidenotes. No
JS branch, no separate mobile component; the same markup re-flows.

**Resolved vs. missing is a tone, set on the note kind, not separate widgets.**
A `[^N]` that resolves against the finding's citations → a `†` amber "source"
note (`--color-token-text`) linking to the chunk; an unresolved `[^N]` → a `‡`
danger-toned "no longer available" note (`--color-danger-text`). External
`[^url:…]`/`[^doc:…]` markers ride the same `†` "source" path (link / muted
ref). One ordered pass over the body builds `bodyHtml` and the `notes` array
together so gutter order always matches reading order.

**Markup is a stable contract for the R3 icon pass (#592).** The
`cite-ref`/`cite-ref--source`/`cite-ref--gone` classes and the
`margin-note__dagger` badge are deliberately semantic and untouched by styling
churn, so #592 can splice a Pixelarticons glyph into the dagger anchor without
reaching into this component's render logic.

CSS lives in a self-contained `R4 · Finding-detail ledger treatment` section at
the end of `app.css` (owns only `.ledger*`, `.drop-cap-body`, `.cite-ref*`,
`.cite-notes`, `.margin-note*`, and a `.split:has(.ledger)` width override) so
it doesn't collide with the other R-series players editing the same file. No
schema change; web-only.
