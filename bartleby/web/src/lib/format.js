// Strip a trailing extension (e.g. .pdf) so a file name reads as a title when
// no summary-derived title exists. Returns falsy input unchanged so callers can
// chain a further `?? fallback`. Shared by the list, detail, and search views.
export function stripExt(name) {
  return name ? name.replace(/\.[^.]+$/, '') : name;
}

// Render `<count> <noun>` with naive English pluralization and a comma-grouped
// count (1234 → "1,234"). Pass `plural` when the noun isn't just `singular + 's'`
// (e.g. pluralize(n, 'match', 'matches')).
export function pluralize(count, singular, plural = `${singular}s`) {
  const shown = typeof count === 'number' ? count.toLocaleString('en-US') : count;
  return `${shown} ${count === 1 ? singular : plural}`;
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
