// Search source kinds. DEFAULT_KINDS is the set searched when the user hasn't
// ticked any box; ALL_KINDS is the full set of checkboxes. Shared by the search
// form (client) and the search loader (server) so the two can't drift.
export const DEFAULT_KINDS = ['documents', 'findings', 'images'];
export const ALL_KINDS = ['documents', 'summaries', 'findings', 'images'];
