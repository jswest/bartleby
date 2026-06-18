// Strip a prefixed entity id (`<type>:<int>`) to a bare integer for SQLite binding.
// Tolerant of bare integers and numeric strings (DB-sourced lists emit those).
// Returns NaN on anything unparseable — callers decide whether to 404 or skip.

const PREFIXED = /^(?:chunk|document|finding|image|tag|summary):(\d+)$/;

export function bareId(value) {
  if (typeof value === 'number') return Number.isInteger(value) ? value : NaN;
  if (typeof value !== 'string') return NaN;
  const s = value.trim();
  const m = PREFIXED.exec(s);
  if (m) return Number(m[1]);
  return /^\d+$/.test(s) ? Number(s) : NaN;
}
