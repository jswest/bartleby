// Strip a trailing extension (e.g. .pdf) so a file name reads as a title when
// no summary-derived title exists. Returns falsy input unchanged so callers can
// chain a further `?? fallback`. Shared by the list, detail, and search views.
export function stripExt(name) {
  return name ? name.replace(/\.[^.]+$/, '') : name;
}

// Render `<count> <noun>` with naive English pluralization. Pass `plural` when
// the noun isn't just `singular + 's'` (e.g. pluralize(n, 'match', 'matches')).
export function pluralize(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}
