# `.txt` document sources render as literal escaped text in a `<pre>`, never through `marked`.

The two-pane detail layout (`/documents/[id]`) fell back to the browser's unstyled
native `text/plain` view for `.txt` sources: `SourceViewer`'s `isMarkdown` regex
(`/\.(md|markdown)$/i`) didn't match, so a `.txt` fell through to the raw-file
`<iframe src>` branch. The obvious-looking fix — add `txt` to the `isMarkdown`
regex and route it through the same `marked.parse` path as `.md` — is wrong and
was explicitly rejected in favor of a dedicated literal-text branch. `marked`
runs repo-wide with `breaks: false` (only a DOMPurify sanitize hook is registered
in `+layout.svelte`), so parsing plain text as markdown would collapse single
newlines into run-on paragraphs and misinterpret any stray `#`/`*`/`_`/`___`
sequence as heading/emphasis/hr syntax — exactly the shape of line-structured
metadata dumps and log-like `.txt` corpora this viewer needs to show faithfully.
This is the same hazard already codified elsewhere in the codebase: `format.js`'s
`isMarkdownChunk` and `ResultCard.svelte` both single out extracted/plain text as
needing to stay literal, specifically because running it through `marked` would
mangle stray markdown-looking characters that were never authored as markdown.

So `SourceViewer.svelte` gained a dedicated `isText` predicate
(`/\.(txt|text|log)$/i`) alongside `isMarkdown`, and both feed a shared
`isInline` reactive gate. Inline sources reuse the *same* resolution path
(pre-loaded SSR text if present, else `fetch(src)`) and the *same* sandboxed
`srcdoc` wrapper (`wrapDocument`, CSP `default-src 'none'`) already in place for
markdown (see GH-0567) — but the `srcdoc` body branches on `isMarkdown`: markdown
still goes through `marked.parse`, while text is HTML-escaped (via the shared
`escapeHtml` in `format.js`) and dropped verbatim into a `<pre>`. The shared `pre` CSS rule picked up
`white-space: pre-wrap; overflow-wrap: anywhere;` so long unbroken lines (e.g.
YAML-ish dumps) wrap inside the pane instead of relying on `overflow-x: auto`
alone. The SSR preload in `documents/[id]/+page.server.js` widened past `\.md$`
to also cover `.markdown`/`.txt`/`.text`/`.log` so these get the same
no-client-round-trip treatment as markdown; the prop carrying this text was
renamed from `markdown` to `sourceText` since it now carries either kind.
Consequence: a `.txt` document now renders as styled, wrapped, but strictly
literal text — no markdown interpretation, no unstyled native browser dump.
Frontend-only; no `SCHEMA_VERSION` bump (issue #680).
