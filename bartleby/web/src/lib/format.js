// Clamp a plain-text snippet to at most `maxLen` characters, cutting on a word
// boundary and appending a single ellipsis (…). Handles two truncation cases:
//
//   1. Python's `apply_preview` appends a Unicode '…' when it truncates — strip
//      that first, then recut on a word boundary so mid-word cuts are healed.
//   2. Source text longer than `maxLen` — cut on word boundary, append '…'.
//
// In both cases trailing punctuation is stripped before the ellipsis so source
// punctuation never collides with ours (no "text......", no "particulat……").
// Returns the text unchanged (but trailing punctuation stripped) when it fits
// within `maxLen` and was not Python-truncated.
//
// Operates on PLAIN TEXT only — do not pass markup. B3's highlight pass wraps
// matched terms in <mark> after this helper runs.
export function clampSnippet(text, maxLen = 280) {
  if (!text) return text;
  // Python's apply_preview appends a single Unicode '…' when truncating.
  // Detect and strip it so it never collides with ours.
  const wasPythonTruncated = text.endsWith('…');
  const stripped = wasPythonTruncated
    ? text.slice(0, -1).replace(/[\s\p{P}]+$/u, '')
    : text.replace(/[\s\p{P}]+$/u, '');
  // Short enough and not mid-word-truncated: return clean (no ellipsis).
  if (stripped.length <= maxLen && !wasPythonTruncated) return stripped;
  // Recut on word boundary and append a single ellipsis.
  const effectiveMax = Math.min(maxLen, stripped.length);
  const cut = stripped.slice(0, effectiveMax);
  const wordBoundary = cut.search(/\S\s+\S*$/);
  const trimmed = (wordBoundary > 0 ? cut.slice(0, wordBoundary + 1) : cut)
    .replace(/[\s\p{P}]+$/u, '');
  return trimmed + '…';
}

// Render a date range as "YYYY-MM-DD – YYYY-MM-DD", collapsing to a single
// date when start == end (e.g. a one-day corpus or a single-day session span).
// Accepts either two separate strings or a single object with `.min`/`.max`.
// Returns null when `a` is falsy so callers can use {#if formatDateRange(...)}.
export function formatDateRange(a, b) {
  const lo = typeof a === 'object' && a !== null ? a.min : a;
  const hi = typeof a === 'object' && a !== null ? a.max : b;
  if (!lo) return null;
  return lo === hi ? lo : `${lo} – ${hi}`;
}

// Filename slug: lowercase, non-alphanumerics collapsed to a single '-', with
// leading/trailing '-' trimmed. Mirrors `_slug()` in bartleby/commands/finding.py
// so a finding's exported filename is stable across the CLI and both web
// export paths (Download .md, Save as HTML).
export function slugify(title) {
  const s = (title || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return s || "finding";
}

// Strip a trailing extension (e.g. .pdf) so a file name reads as a title when
// no summary-derived title exists. Returns falsy input unchanged so callers can
// chain a further `?? fallback`. Shared by the list, detail, and search views.
export function stripExt(name) {
  return name ? name.replace(/\.[^.]+$/, '') : name;
}

// Escape the HTML metacharacters so untrusted text renders literally when
// interpolated into an HTML string — the source viewer's plain-text `<pre>` and
// the findings citation renderer both drop text into markup. `&` is replaced
// first (via the single character-class pass) so the others aren't re-escaped.
export function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

// Render `<count> <noun>` with naive English pluralization and a comma-grouped
// count (1234 → "1,234"). Pass `plural` when the noun isn't just `singular + 's'`
// (e.g. pluralize(n, 'match', 'matches')).
export function pluralize(count, singular, plural = `${singular}s`) {
  const shown = typeof count === 'number' ? count.toLocaleString('en-US') : count;
  return `${shown} ${count === 1 ? singular : plural}`;
}

// Split a plain-text snippet into segments for client-side term highlighting.
// Returns an array of {text: string, highlight: boolean} objects whose `text`
// values, concatenated, equal the original snippet. Matched spans are flagged
// so the caller can wrap them in <mark> without any raw-HTML injection.
//
// Matching is case-insensitive; each whitespace-separated token in `query` is
// treated as an independent term (same tokenisation as the FTS search leg).
// Only used on PLAIN-TEXT snippets — markdown snippets go through `marked` and
// keep their own emphasis, so we leave them unmarked rather than injecting
// <mark> mid-parse.
//
// Returns [{text, highlight: false}] when query is absent/empty so callers
// can always iterate the return value without a guard.
export function highlightTerms(text, query) {
  if (!text) return [];
  if (!query || !query.trim()) return [{ text, highlight: false }];

  // Build one regex that matches any token in the query (case-insensitive,
  // substring — "cat" also highlights inside "category"). Escape special regex
  // chars in each token so a query like "C++ pointer" doesn't blow up.
  const tokens = query.trim().split(/\s+/).filter(Boolean);
  const escaped = tokens.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const re = new RegExp(`(${escaped.join('|')})`, 'gi');

  const segments = [];
  let last = 0;
  for (const match of text.matchAll(re)) {
    if (match.index > last) {
      segments.push({ text: text.slice(last, match.index), highlight: false });
    }
    segments.push({ text: match[0], highlight: true });
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    segments.push({ text: text.slice(last), highlight: false });
  }
  return segments;
}

// Whether a chunk's text is reliably markdown-authored — so it should render
// through `marked` — versus extracted/plain text that must stay literal (running
// it through marked would mangle stray *, _, # chars). Agent/model output
// (findings, summaries) and a Docling pipe table (sec_table) and a Markdown
// source file (.md) are markdown; everything else is literal. Shared by the
// search/scan cards and the /chunks view so the rule lives in one place.
export function isMarkdownChunk({ source_kind, content_type, file_name }) {
  return (
    source_kind === 'finding' ||
    source_kind === 'summary' ||
    content_type === 'sec_table' ||
    (file_name?.toLowerCase().endsWith('.md') ?? false)
  );
}
