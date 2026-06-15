# Marginalia citations: dagger-aligned sidenotes + right-pane note click (issue #631)

> Source: [#631](https://github.com/jswest/bartleby/issues/631)

Two UX complaints against the R4 gutter citations, both addressed here.

## Change 1 — Dagger-aligned sidenotes (the "not in line" fix)

**The problem.** The R4 decision (#593) deliberately chose a simple flex-column
gutter — notes stack in reading order but have no geometric relationship to their
inline dagger anchors. The user found this disorienting: a note for text near the
top of the page might render halfway down the gutter, with no visual tie except
the `†N` ordinal.

**The decision: JS-measured absolute positioning with a downward de-overlap
pass.** A `layoutNotes()` function runs after mount, after every reactive update
to `notes`/`bodyHtml` (via `tick()` so the DOM is settled), and on resize (via a
`ResizeObserver` on the prose body column). For each ordinal N it:

1. Queries `.cite-ref[data-note="N"]` in the prose and reads its top offset
   relative to the `.cite-notes` aside (using `getBoundingClientRect` + `scrollY`
   for sub-pixel accuracy across scroll positions).
2. Assigns that top to the matching `.margin-note[data-note="N"]` as
   `position: absolute; top: Npx`.
3. Sweeps top→bottom with a `runningBottom` cursor: if a note's desired top
   would overlap the one above, it is nudged down to `runningBottom`. A fixed
   `NOTE_GAP` (8 px) keeps adjacent notes from touching.
4. Sets the aside's `height` to `runningBottom` so it doesn't collapse under its
   absolutely-positioned children.

`.cite-notes` becomes `position: relative` (the positioning context); individual
`.margin-note` elements receive inline `position: absolute` + `top`.

**Why not CSS-only?** Tufte-style sidenote positioning requires knowing the
rendered vertical offset of each inline anchor — information that only exists
after layout. CSS `position: sticky` and `float` both fight the grid column and
produce unpredictable overlaps when two daggers land on adjacent lines. A pure
CSS approach would require one `<aside>` per note, which breaks the current
single-gutter structure and the `scrollIntoView` click handler.

**Narrow-screen fallback.** The `@media (max-width: 56rem)` block (pre-existing
from #593) collapses the ledger column to one column, folding the gutter below
the prose as end-notes. At this breakpoint the layout function no-ops: it checks
`citesAside.offsetWidth <= 0` (the grid track gives the aside zero width when
the column is collapsed) and returns early. When the aside is narrow the
`resetNoteLayout()` helper removes inline `position`/`top` styles so the CSS
`position: static !important` rule in the media query takes full effect. The
`†N` ordinals still tie each inline anchor to its end-note — no information
loss.

## Change 2 — Click a margin note to open the source in the right pane

**The problem.** The corpus-chunk margin note (`{:else}` branch) previously did
`goto('/chunks/${note.chunkId}')` — navigating the whole page away. The user
wanted to either open the chunk in the right pane (same as clicking the inline
dagger) or navigate to the chunk page, with the former as the primary action.

**The decision: split into primary label + secondary icon affordance.**

- **Primary (clicking the note label):** calls `activate(note.chunkId)` — the
  same function the inline dagger's click handler calls. This loads the document
  in the right-pane `SourceViewer` without any navigation. The existing `.active`
  highlight (`class:active={note.active}`) is preserved.
- **Secondary (↗ icon):** a small `<a>` rendered after the label using the
  existing `CHUNK_ICON` from `$lib/icons.js` (the "open in context" pixel glyph
  already used on result cards). Clicking it calls `goto('/chunks/${note.chunkId}')`.
  Styled at 60% opacity, brightening to 100% on hover — visually minor against
  the amber label, matching the retro pixel-art aesthetic.

The outer element changed from `<button>` to `<span class="margin-note__source-row">`,
containing the `<button>` label and `<a>` icon as siblings. The `data-chunk-id`
attribute stays on the `<button>` so the existing dagger-click handler
(`.querySelector("[data-chunk-id]")` → `activate(...)`) continues to work without
modification.

The `url`, `doc`, and `gone` branches are unchanged.
