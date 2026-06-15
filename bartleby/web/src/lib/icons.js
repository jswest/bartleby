// Pixelarticons (https://pixelarticons.com, MIT) — a 1-bit pixel-art glyph set.
// We DON'T pull in the npm package or an icon framework: each glyph here is the
// raw <svg> copied verbatim from the Pixelarticons source and inlined as a
// string, so the bundle carries only the handful of icons we actually use (no
// runtime icon dependency). This matches the existing pattern: ResultCard drops
// the markup via {@html}, and the finding viewer splices it into citation HTML
// it builds by hand.
//
// Every glyph is a 24×24, fill="currentColor" path, so it inherits the
// link/text color at each site (nav shell-text, citation amber, card ink). The
// shared `pixel` class carries the sizing + `image-rendering: pixelated` (so
// scaled glyphs stay crisp 1-bit squares) — see the R3 block in app.css.
//
// Helper: stamp the shared class + a11y attrs onto a raw Pixelarticons body so
// every export reads identically.
const icon = (paths, { size = 16 } = {}) =>
  `<svg class="pixel" viewBox="0 0 24 24" width="${size}" height="${size}"` +
  ` fill="currentColor" aria-hidden="true" focusable="false">${paths}</svg>`;

// Nav: Search (magnifying glass).
export const SEARCH_ICON = icon(
  '<path d="M22 22h-2v-2h2v2Zm-2-2h-2v-2h2v2Zm-6-2H6v-2h8v2Zm4 0h-2v-2h2v2ZM6 16H4v-2h2v2Zm10 0h-2v-2h2v2ZM4 14H2V6h2v8Zm14 0h-2V6h2v8ZM6 6H4V4h2v2Zm10 0h-2V4h2v2Zm-2-2H6V2h8v2Z"/>',
);

// Findings (sparkle) — agent output, the same signal the mint cards carry.
export const FINDINGS_ICON = icon(
  '<path d="M11 1h2v4h-2zm0 22h2v-4h-2zM9 5h2v4H9zm0 14h2v-4H9zm4-14h2v4h-2zm0 14h2v-4h-2zM5 9h4v2H5zm14 0h-4v2h4zM1 11h4v2H1zm22 0h-4v2h4zM5 13h4v2H5zm14 0h-4v2h4z"/>',
);

// Documents (file with text lines) — source material with summaries.
export const DOCUMENTS_ICON = icon(
  '<path d="M6 4H4v16h2zm10-2H6v2h10zm4 4h-2v14h2zm-2 14H6v2h12zM16 4h2v2h-2zm-4 0h2v6h-2z"/><path d="M12 8h6v2h-6zm-4 8h8v2H8zm0-4h8v2H8zm0-4h2v2H8z"/>',
);

// Tags (hash) — Pixelarticons has no tag glyph; tags read as #docket / #filer,
// so the hash is the right semantic stand-in.
export const TAGS_ICON = icon(
  '<path d="M9 3h2v5H9zm6 0h2v5h-2zm-7 7h2v4H8zm6 0h2v4h-2zm-7 6h2v5H7zm6 0h2v5h-2zM3 8h18v2H3zm0 6h18v2H3z"/>',
);

// "Open in context" — an arrow leaving a box. Next to every chunk reference
// (search/scan cards, finding citation chips) it links to /chunks/<id>.
// Replaces the lone hand-rolled ↗ glyph; the link still opens in a new tab.
export const CHUNK_ICON = icon(
  '<path d="M11 5H5v2h6V5ZM5 7H3v12h2V7Zm12 12H5v2h12v-2Zm2-6h-2v6h2v-6Zm-8 0H9v2h2v-2Zm2-2h-2v2h2v-2Zm2-2h-2v2h2V9Zm2-2h-2v2h2V7Zm2-2h-2v2h2V5Zm2-2h-2v8h2V3Z"/><path d="M21 3h-8v2h8V3Z"/>',
  { size: 11 },
);
