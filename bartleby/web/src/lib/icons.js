// Inline SVG markup, shared so the same glyph reads identically wherever a chunk
// is linked. Emitted as a raw string: ResultCard drops it via {@html}, and the
// finding viewer splices it into the citation-chip HTML it builds by hand. Uses
// currentColor so it inherits the link/text color at each site.

// "Open in context" — a small arrow leaving a box. Next to every chunk
// reference (search/scan cards, finding citation chips) it links to /chunks/<id>.
export const CHUNK_ICON =
  '<svg class="icon" viewBox="0 0 16 16" width="11" height="11" aria-hidden="true" focusable="false">' +
  '<path d="M9 3h4v4M13 3 8 8M11 9.5V12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h2.5" ' +
  'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
  '</svg>';
