# `bartleby serve` writes for the first time — annotation endpoints only, through one writable better-sqlite3 handle.

Until #734 the web app was strictly read-only: `src/lib/server/db.js` opened the
project DB with `readonly: true` and there was no `POST`, no form action, nothing
that could change a corpus. Finding annotations (#689) are the first thing a human
needs to *write* from the browser — the point of the feature is reading a finding
and noting, right there, that a sentence is wrong — so serve now writes.

**Scope of the write surface.** Exactly two endpoints:
`POST /findings/[id]/annotations` and
`DELETE /findings/[id]/annotations/[annotation_id]`. They call
`insertAnnotation`/`deleteAnnotation` in `src/lib/server/annotations.js`, which is
the only importer of `getWritableDb()` (a second memoized handle in `db.js`,
`readonly: false`, `busy_timeout = 5000`, `foreign_keys = ON`, opened lazily on
the first write). Every page, query, and the HTML export keep using the
untouched read-only `getDb()`. Widening the write surface means a new decision,
not a new import of `getWritableDb`.

**Why a writable handle rather than spawning a skill script** (the path search
already uses via `skill.js`):

- *Authorship is honest.* Skill scripts are an agent's surface: they resolve a
  session and stamp it. A note typed in the browser has no session, so the
  web writes `is_human_author = 1`, `session_id = NULL` — what #689's contract
  pins for human notes — instead of borrowing or minting a session to satisfy a
  script.
- *No memory-wall entanglement.* Skill-side annotation reads/writes route
  through `assert_findings_accessible` (#689) because an agent session may have
  memory off. A human at their own corpus has no memory policy; going through
  the skill would either inherit a wall that doesn't apply or need a bypass flag.
- *No Python in the loop.* A direct insert is one transaction in-process —
  no subprocess spawn per note, no `uv`/Python environment dependency for a
  write the web can validate itself.

**What keeps the two writers consistent.** The DDL is pinned in #689 and owned
by `bartleby/db/schema.py`; the web never creates or migrates tables. The
anchor check is `locateAnchor` in `src/lib/annotations.js`, a line-for-line
mirror of `bartleby/db/annotations.py:locate_anchor` (pinned by
`src/lib/annotations.test.js`, `npm test`). Inserts validate the finding exists
(404), the anchor locates in the current body (400 `anchor_not_found`), and the
chunk exists (400), inside the same transaction as the insert. Errors are JSON
`{error, code}`, like the skill scripts.

**Note text is plain text**, rendered through Svelte's escaping (and
`escapeHtml` in the export), not markdown — so there is nothing for the
DOMPurify hook (GH-0123) to sanitize and no second rendering path to keep safe.
