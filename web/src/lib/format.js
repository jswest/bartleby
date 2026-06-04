// Strip a trailing extension (e.g. .pdf) so a file name reads as a title when
// no summary-derived title exists. Shared by the document list and detail views.
export function stripExt(name) {
  return name.replace(/\.[^.]+$/, '');
}
