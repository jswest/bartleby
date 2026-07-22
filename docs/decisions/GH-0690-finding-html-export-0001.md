# "Save as HTML" exports a finding as one self-contained file — fonts, sources, and margin-note behavior all embedded, no server.

A new route, `GET /findings/[id]/export.html`, assembles a standalone HTML
document that reproduces the live `/findings/[id]` view with everything it
needs baked in: the three self-hosted font families as base64 `data:` URIs
inside `@font-face`, every cited source embedded (PDF/image as base64 `data:`
URIs, `.txt`/`.log` as literal escaped `<pre>` per GH-0680, `.md` rendered the
same way `SourceViewer` renders it live), and a small vanilla-JS re-expression
of the margin-note layout + note→viewer click wiring so the page works with no
Svelte runtime, no fetches, and — per the issue — no network at all. The
finding page's toolbar gained a third button (`Save as HTML`, beside `Copy as
Markdown`/`Download .md`) that navigates to the endpoint; since the endpoint
sets `Content-Disposition: attachment`, the browser downloads it instead of
navigating away.

**Shared logic, not a copy-paste fork.** The citation-marker → dagger-note
substitution (`[^chunk:N]`/`[^finding:N]`/`[^url:…]`/`[^doc:…]` → ¶/§/†/‡ +
gutter note) used to live entirely inside `+page.svelte`'s `renderBody()`. It's
now `$lib/citations.js`'s `substituteCitations()` — a pure function with no
Svelte, DOM, or `marked` dependency — imported by both `+page.svelte` (which
still owns its own `marked.parse()` call, unchanged) and the export builder.
Two more pieces moved to shared modules for the same reason: the sandboxed-
iframe document wrapper + inline viewer CSS (`SourceViewer.svelte`'s
`wrapDocument`/`VIEWER_CSS`) became `$lib/viewerDoc.js`, and the
extension→MIME table (`files/[document_id]/+server.js`'s `MIME_BY_EXT`) became
`$lib/server/mime.js`. Both the live route and the export now import these
instead of each holding its own copy. `+page.svelte` and `SourceViewer.svelte`
are otherwise untouched — no behavior change to the live view.

**Sanitization: an isolated `Marked` instance, not "DOMPurify needs a DOM."**
The issue's open question here turned out to be moot: `isomorphic-dompurify`
is already a dependency and already runs server-side in this exact app — the
global `marked` singleton's DOMPurify-sanitize + target/rel hook is registered
by `+layout.svelte`'s module-context script, and that code runs during SSR of
every page, not just in the browser. So there's nothing awkward about running
DOMPurify in Node here; it's proven. The wrinkle is only that the export route
is a `+server.js` endpoint, not a page component — it never imports
`+layout.svelte`, so it can't rely on that hook having been registered (it
would only be "registered" as a side effect of some other page having
happened to render first in the same process, which is not something to
depend on). The fix is a private `new Marked()` instance in
`exportHtml.js` carrying the identical hook, used for both the finding body
and any embedded `.md` source — sanitized unconditionally, independent of
import order, no new dependency.

**The three open questions, resolved as directed:**
1. *PDF viewing*: embedded as `data:application/pdf;base64,...#page=N&navpanes=0`
   in an iframe (desktop Chrome/Firefox render this inline) plus a fallback
   `<a href="{same data URI}" download="{original filename}">` link under the
   viewer, so a browser that won't render the PDF inline can still save it.
2. *External/cross-finding citations*: a `[^url:…]` note keeps its real
   hyperlink (`target="_blank" rel="noopener noreferrer"`, since it's now
   leaving a local file rather than navigating the SPA); a `[^doc:…]` note was
   already inert live (a plain label, no link) and stays that way; a
   `[^finding:N]` note drops its `/findings/N` link entirely in the export —
   `renderMarginNoteHtml()` in `exportHtml.js` deliberately ignores the `href`
   that `substituteCitations()` still returns (used by the live page) and
   renders inert text instead, since the other finding isn't in this file.
   One more departure follows the same logic: a resolved corpus-chunk note
   drops the "open chunk" icon link (`/chunks/N` doesn't exist standalone
   either) — the label is still clickable to load the source in the viewer,
   just without the secondary navigation affordance.
3. *Image chunks* (`source_kind='image'`): embedded as the archived `.jpg`
   itself, not the parent document. `resolveSources()` in `queries.js` maps an
   image citation to its *parent document's* `document_id`/`file_name` (for the
   live page's citation label and "open in the PDF at this page" link — that
   mapping is untouched), but the actual bytes worth embedding are the
   extracted image's own file. `getCitations()` now also carries `source_kind`/
   `source_id` through (image chunks: `source_id` is the `image_id`), and a new
   `getImageFilePath(imageId)` in `queries.js` resolves it to `images.file_path`.

**Source embeds are deduplicated by document/image id, not by citation.**
Several citations can point at the same document (different pages) or the
same image. Embedding a fresh copy of the PDF's base64 data per citation would
turn "large" into "citation-count × file-size" — a 50-page PDF cited eight
times would repeat its full bytes eight times over. Instead `exportHtml.js`
builds one `SOURCES` map keyed by `doc:<document_id>` / `image:<image_id>` and
a small `CITATIONS` map from `chunk_id` → `{key, page}`; the embedded runtime
JS looks up the shared payload and (for PDFs) appends the per-citation page
fragment at click time. The file is still explicitly allowed to be massive —
this just keeps it proportional to the number of *unique* cited sources, which
is the trade-off actually intended.

**The CSS (and fonts) are a hand-copied snapshot, not a shared import.** Fonts
must be captured as base64 at request time regardless — there's no way to
`@import` a `data:` URI from the live `fonts.css`. Given that, `exportHtml.js`
also carries a curated subset of `app.css`'s rules (tokens, base typography,
`.surface`/`.markdown-body`, the R4 ledger/margin-note/cite-ref treatment, the
R3 dagger bookmark, `SourceViewer`'s placeholder states) as one literal
`EXPORT_CSS` string, rather than parsing/transforming `app.css` at request
time. This is an accepted trade-off, not an oversight: a future visual change
to `app.css` won't automatically reach the export until this string is updated
by hand. Given the fonts already require a hand-triggered re-snapshot on any
font change, this isn't a new failure mode — just the same one, extended to
cover the rest of the visual treatment.

**Fonts are read from `static/fonts/` via a `process.cwd()`-relative path.**
`bartleby serve` (`commands/serve.py`) always launches the vite *dev* server —
never the `adapter-node` production build — out of `~/.bartleby/serve`, which
`_sync_web()` populates by copying/symlinking this app's `static/` tree
verbatim alongside `src/`, then `os.chdir()`s there before `exec`. So
`static/fonts/*.woff2` is a stable path relative to `process.cwd()` in the one
way this app is actually run; `exportHtml.js` reads the six font files from
there rather than trying to resolve a path relative to its own (post-bundling,
potentially-rolled-up) module location, which would not survive a production
build's chunk reshuffling.

**Endpoint shape.** `routes/findings/[id]/export.html/+server.js` — the `.html`
segment gives the route a natural, memorable URL; the `Content-Disposition:
attachment; filename="<slug>.html"` header (slug matching `_slug()` in
`bartleby/commands/finding.py` / the web `downloadMarkdown()`) is what actually
drives the download, not the URL shape. No CLI counterpart was added (the
issue treats that as optional/deferred); the web button is the whole ask here.

**Verification.** No automated test seam exists for exercising a SvelteKit
route end-to-end from the Python suite, so this was verified by hand: a
throwaway sandbox (`BARTLEBY_HOME` pointed at a temp dir, never a real
project) with a project seeded directly through the typed chunk-insert
helpers — a 3-page generated PDF, a `.txt` file (including a literal
`</script>` in its body, to probe the JSON-embedding escape), a small JPEG
registered as an extracted image chunk, and a finding citing all three plus a
`[^url:…]`, a `[^doc:…]`, a `[^finding:999]`, and a dangling `[^chunk:999999]`
— then the vite dev server was run against that sandbox only and the endpoint
hit directly. The resulting file was checked for: all three `@font-face`
families as base64 `data:font/woff2`, an embedded `data:application/pdf`, an
embedded `data:image/jpeg` (correctly named after the archived `figure.jpg`,
not the parent PDF), the literal (unmangled) `.txt` content including its
`</script>` probe text, all four dagger glyphs present, zero `http://` or bare
`https://` references other than the one `[^url:…]` citation's own href, no
`/chunks/` or cross-finding `/findings/999` links, and balanced `<script>`
tags. `uv run pytest` (untouched — no Python changed) and `npm run build`
both pass.
