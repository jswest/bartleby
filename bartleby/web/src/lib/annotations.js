// Text-quote anchors for finding annotations (#689). Pure string logic, shared
// by the annotation endpoints (validation, `anchor_found`) and the finding page
// (mapping a selection to an anchor, placing highlights). No DOM, no DB —
// the page supplies the render round-trip.
//
// An anchor is a W3C-style text-quote selector over the RAW markdown body:
// `exact` is a verbatim substring; optional `prefix`/`suffix` pick out one
// occurrence when `exact` repeats.

// Mirrors bartleby/db/annotations.py:locate_anchor exactly — the pinned
// contract in #689. Returns [start, end] of the first occurrence of `exact`
// (if prefix/suffix are given, the first occurrence they bracket), or null.
// An empty prefix/suffix counts as absent; an empty `exact` is invalid.
export function locateAnchor(body, exact, prefix = null, suffix = null) {
  if (!exact) throw new Error('anchor exact must be non-empty');
  let start = body.indexOf(exact);
  while (start !== -1) {
    const end = start + exact.length;
    const prefixOk = !prefix || (start >= prefix.length && body.startsWith(prefix, start - prefix.length));
    const suffixOk = !suffix || body.startsWith(suffix, end);
    if (prefixOk && suffixOk) return [start, end];
    start = body.indexOf(exact, start + 1);
  }
  return null;
}

// Spans of the raw body an anchor boundary must not fall strictly inside:
// citation markers `[^…]` and link/image targets `](…)`. Mirrors the Python
// write-time rule (#689).
// Scanned as two separate passes (not one alternation) so overlapping spans such as
// `[^a](b)` both count — this mirrors `splits_protected_span` in db/annotations.py.
const MARKER_SPANS = [/\[\^[^\]]*\]/g, /\]\([^)]*\)/g];

// True when `start` or `end` falls STRICTLY inside a marker/link-target span
// of `body` (span.start < pos < span.end). An anchor that fully contains a
// marker — or merely touches one — is fine.
export function splitsMarker(body, start, end) {
  for (const re of MARKER_SPANS) {
    for (const m of body.matchAll(re)) {
      const s = m.index, e = m.index + m[0].length;
      if ((s < start && start < e) || (s < end && end < e)) return true;
    }
  }
  return false;
}

// Private-use sentinels bracketing a raw span, so the span can be traced
// through markdown rendering to its offsets in the rendered text.
const OPEN = '';
const CLOSE = '';

export function bracketSpan(body, start, end) {
  return body.slice(0, start) + OPEN + body.slice(start, end) + CLOSE + body.slice(end);
}

// Given the rendered text of a bracketSpan()-ed body and the rendered text of
// the plain body, the [start, end) the span occupies in the plain rendered
// text — or null when the sentinels did not survive rendering intact, the
// rendering otherwise changed, or the span renders to nothing.
export function renderedSpan(bracketedText, plainText) {
  const start = bracketedText.indexOf(OPEN);
  const close = bracketedText.indexOf(CLOSE);
  if (start === -1 || close < start) return null;
  const end = close - 1;
  const stripped = bracketedText.slice(0, start) + bracketedText.slice(start + 1, close) + bracketedText.slice(close + 1);
  if (stripped !== plainText || end <= start) return null;
  return [start, end];
}

// Map a selection in the RENDERED text to a raw-body anchor, from rendered
// context: `quote` must occur verbatim in the raw body; prefix/suffix are the
// rendered text around the selection, trimmed (longest first) until the
// anchor locates in the raw body. A candidate counts only if it does not split
// a marker/link target and `placeSpan(rawStart, rawEnd)` — the caller's render
// round-trip — lands exactly on the selection. Null when nothing verifies; the
// caller then offers a whole-finding note rather than a fabricated anchor.
const CONTEXT_LENGTHS = [32, 16, 8, 4, 0];

export function anchorFromRendered(body, rendered, selStart, quote, placeSpan) {
  const selEnd = selStart + quote.length;
  const pairs = CONTEXT_LENGTHS.flatMap((p) => CONTEXT_LENGTHS.map((s) => [p, s]))
    .sort((a, b) => b[0] + b[1] - (a[0] + a[1]));
  for (const [p, s] of pairs) {
    const prefix = rendered.slice(Math.max(0, selStart - p), selStart) || null;
    const suffix = rendered.slice(selEnd, selEnd + s) || null;
    const hit = locateAnchor(body, quote, prefix, suffix);
    if (!hit || splitsMarker(body, ...hit)) continue;
    const span = placeSpan(...hit);
    if (span && span[0] === selStart && span[1] === selEnd) return { exact: quote, prefix, suffix };
  }
  return null;
}
