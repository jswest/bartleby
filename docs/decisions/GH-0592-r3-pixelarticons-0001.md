# Pixelarticons via inlined SVG strings, not a runtime icon framework (issue #592)

> Source: [#592](https://github.com/jswest/bartleby/issues/592)

R3 adopts **[Pixelarticons](https://pixelarticons.com)** — a 1-bit pixel-art
glyph set, **MIT-licensed** — as the redesign's icon vocabulary, replacing the
lone hand-rolled chunk-↗ SVG and adding consistent glyphs to nav links, card
eyebrows, and the finding-detail citation daggers. The hand-rolled pixels read
as "adorable but a little messy"; an extant set that plays nice with the
Iosevka/Doto chrome was wanted, and Pixelarticons' square 1-bit glyphs are that.

**The set is inlined as raw `<svg>` strings in `lib/icons.js`, NOT pulled in as
an npm package or icon framework.** Neither `pixelarticons` nor
`@iconify-json/pixelarticons` is a dependency. We copy the handful of glyphs we
actually use (`search`, `sparkle`, `file-text`, `hash`, `external-link`,
`bookmark`) verbatim from the Pixelarticons source and stamp them with a shared
`icon()` helper. Rationale: this extends the pattern that already existed
(`CHUNK_ICON` was a `{@html}`'d SVG string), keeps the bundle to only the icons
in use with zero tree-shaking ceremony and no runtime icon component, and avoids
an Iconify build-plugin or a `<Icon>` component for what is six static glyphs.
Each glyph is `fill="currentColor"` so it inherits its site's text color — nav
shell-text, card ink, citation amber — exactly like the prior pattern. The
shared `.pixel` class (in the new `R3` block at the end of `app.css`) carries
`image-rendering: pixelated` so any scaled glyph stays crisp 1-bit squares.

**Glyph mapping.** Nav: `search` → Search, `sparkle` → Findings (the agent-output
signal the mint cards carry), `file-text` → Documents (source-with-summaries),
`hash` → Tags (Pixelarticons ships no tag glyph; tags read as `#docket`/`#filer`,
so the hash is the right semantic stand-in). Card eyebrows reuse
`sparkle`/`file-text` for the home Findings/Documents tiles and the
`ResultCard` source-kind eyebrow (finding → sparkle, else file-text).
`external-link` replaces the hand-rolled chunk-↗; the chunk link's
href/title/aria/same-tab behavior is untouched (R4's markdown citation path,
which opens in a new tab via the `marked` hook, no longer splices `CHUNK_ICON`).

**Citations: a CSS mask, no R4 render change.** R4 (#593) left
`margin-note__dagger` and `cite-ref` as a stable contract. Rather than edit
R4's render logic (which would risk the †/ordinal tie-back), the dagger gets a
small Pixelarticons `bookmark` painted by a `::before` pseudo via CSS `mask`
(the glyph is a `--pixel-bookmark` data-URI; mask uses alpha, so the dagger's
`currentColor` — amber for source, danger for gone — shows through). The `†N`/
`‡N` text the tie-back depends on is unchanged.

**Scope discipline.** All icon CSS lives in a self-contained
`R3 · Pixelarticons 1-bit icon set` block at the end of `app.css`; it adds
nothing to R1's `:root` tokens (only a `--pixel-bookmark` mask URL) and touches
none of the two-grounds rules. The nav-icon sizing respects B1's (#595) wrap
math — glyphs are fixed squares and shrink to 13px on the ≤480px row, so all
four icon+label links still fit 375px with no horizontal scroll. No schema
change; web-only.
