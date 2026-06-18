# Marginalia polish: glyph rotation, divider removal, link underline (issue #666)

> Source: [#666](https://github.com/jswest/bartleby/issues/666)

## Overview

Three visual tweaks to the finding-detail marginalia (right-hand citation gutter).
All changes are in `bartleby/web/src/routes/findings/[id]/+page.svelte` and
`bartleby/web/src/app.css`.

## Decision 1 — glyph rotation (supersedes GH-0654 and GH-0642)

GH-0642 assigned `†` to corpus chunks and `§` to external `[^url:…]`/`[^doc:…]`
citations; GH-0654 then assigned `¶` to finding-to-finding links. That combined
assignment is now revised so the glyphs better match typographic convention and
visual weight:

| Scheme | Before | After (#666) | Note |
|---|---|---|---|
| `[^chunk:N]` resolved | `†` | `¶` | pilcrow — the primary corpus citation glyph |
| `[^chunk:N]` gone | `‡` | `‡` | unchanged — double-dagger for missing |
| `[^finding:N]` | `¶` | `§` | section sign / silcrow |
| `[^url:…]` / `[^doc:…]` | `§` | `†` | dagger — external reference |

The `kind`-based CSS classes (`cite-ref--source`, `cite-ref--finding`,
`cite-ref--gone`, `margin-note--{source,finding,gone}`) are unchanged — they are
semantic, not glyph-based. After this rotation, resolved corpus chunks (`¶`) and
external url/doc references (`†`) both fall under `cite-ref--source` / amber
color, which is intentional: the glyph alone distinguishes them, and a separate
CSS kind for external would add complexity without visual benefit.

The `marker()` function builds its inline tooltip from `note.dagger` and the kind
(`"§ finding · title"`, `"¶ source · filename"`, etc.), so tooltips update
automatically with the new glyphs.

**Rationale:** pilcrow (¶) is the strongest paragraph/body-text glyph and suits
the primary corpus-chunk role; dagger (†) is the classic footnote opener and suits
the external/link role; section sign (§) is a natural fit for internal
cross-references between findings.

## Decision 2 — remove the body↔marginalia divider

`.cite-notes` previously had `border-left: 1px solid var(--color-rule)` drawing a
vertical rule between the prose column and the note gutter. This is removed — the
column gap (`var(--space-xl)`) provides sufficient visual separation without an
explicit rule.

The narrow-screen `@media (max-width: 56rem)` block had a companion
`border-left: 0` override that was only there to cancel the base rule. With the
base rule gone, that override is redundant and is dropped. The narrow-screen
`border-top` (top rail when notes stack below prose) is retained.

## Decision 3 — marginalia link legibility

`.margin-note__link` (the resolved-source label button) previously showed an
underline only on `:hover` / `.active`. At rest it read as plain amber text,
indistinguishable from the surrounding SOURCE label.

A persistent, low-contrast underline is added at rest:

```css
text-decoration: underline;
text-underline-offset: 2px;
text-decoration-color: color-mix(in srgb, var(--color-token-text) 35%, transparent);
```

On hover/active the `text-decoration-color` strengthens to full
`var(--color-token-text)` so the emphasis is still clear. The amber `color` is
unchanged throughout.

`color-mix` was chosen over a hard-coded translucent hex so the muted underline
tracks the amber token if the token value ever changes.

## Files changed

- `bartleby/web/src/routes/findings/[id]/+page.svelte` — glyph rotation in
  `sourceNote`, `findingNote`, `externalNote`; header comment updated.
- `bartleby/web/src/app.css` — `border-left` removed from `.cite-notes`; redundant
  `border-left: 0` dropped from narrow-screen block; `.margin-note__link` gains
  persistent underline; glyph legend comments updated.
