// Text-quote anchors for finding annotations (#689). Pure string logic, shared
// by the annotation endpoints (validation, `anchor_found`) and the finding page
// (mapping a selection to an anchor, placing highlights). No DOM, no DB.
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

// Start index of the k-th (0-based) occurrence of `needle` in `text`, or -1.
// Steps by one char, like locateAnchor, so overlapping repeats count.
export function nthIndexOf(text, needle, k) {
  let i = text.indexOf(needle);
  while (i !== -1 && k-- > 0) i = text.indexOf(needle, i + 1);
  return i;
}

// How many occurrences of `needle` start before `pos` — i.e. which occurrence
// (0-based) the one at `pos` is.
export function occurrenceIndex(text, needle, pos) {
  let k = 0;
  for (let i = text.indexOf(needle); i !== -1 && i < pos; i = text.indexOf(needle, i + 1)) k++;
  return k;
}

// The minimal anchor that locates the k-th occurrence of `exact` in `body`:
// bare `{exact}` when it is the first occurrence, else the shortest
// prefix/suffix context (doubling from 16 chars) that brackets it. Null when
// `exact` has no k-th occurrence in the raw body — the caller then offers a
// whole-finding note rather than fabricating an anchor. Terminates: with the
// whole remaining body as prefix+suffix only `start` itself can match.
export function anchorFor(body, exact, k) {
  if (!exact) return null;
  const start = nthIndexOf(body, exact, k);
  if (start === -1) return null;
  const end = start + exact.length;
  for (let n = 0; ; n = n ? n * 2 : 16) {
    const prefix = body.slice(Math.max(0, start - n), start) || null;
    const suffix = body.slice(end, end + n) || null;
    if (locateAnchor(body, exact, prefix, suffix)?.[0] === start) {
      return { exact, prefix, suffix };
    }
  }
}
