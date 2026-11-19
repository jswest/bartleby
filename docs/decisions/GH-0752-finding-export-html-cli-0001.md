# `finding export --html` reaches GH-0690's builder through Vite's SSR module loader, and `FONTS_DIR` moves from `process.cwd()` to the module's own file location

`bartleby finding export finding:<N> --html [--out PATH]` (issue #752) writes
the same self-contained file `serve`'s "Save as HTML" button saves (GH-0690),
for an agent that only has the CLI — no `bartleby serve`, no browser. The
proposal was explicit that `buildFindingExportHtml`
(`bartleby/web/src/lib/server/exportHtml.js`) must be reused, not ported to
Python: one builder means the button and the CLI can't drift apart.

**The entry point loads the builder through Vite, not plain `node`.**
`exportHtml.js` itself only has relative imports, but one of its own imports —
`queries.js`'s `import { clampSnippet } from '$lib/format.js'` — uses
SvelteKit's `$lib` alias (`src/lib`), which only Vite's module resolution
understands; a plain `node --experimental-vm-modules` or bare `import()` on
`exportHtml.js` would throw on that alias two hops down the graph. Rather than
touch `queries.js` (owned concurrently by #754) or hand-roll an alias shim,
the new entry point (`bartleby/web/scripts/export-finding-html.js`) uses
Vite's programmatic SSR API — `createServer({ server: { middlewareMode: true
}, appType: 'custom' })` then `server.ssrLoadModule('/src/lib/server/
exportHtml.js')` — the same resolution `vite dev` (what `bartleby serve`
actually runs) performs, just without binding a port. `vite` is already a
`devDependency` the one-time `npm install` in `_sync_web` installs, so this
needed no second install and no new dependency.

**Python resolves the project and the output path; node only builds and
writes.** `commands/finding.py`'s new `_export_html` imports `WEB_SRC`,
`_is_source_checkout`, `_sync_web`, `_needs_install`, and `_require_node`
straight from `commands/serve.py` (reused, not re-implemented) to sync
`~/.bartleby/serve` and run its one-time `npm install` if needed — the exact
steps `serve` itself takes before launching vite — then fails with `serve`'s
own node/npm-missing message if either is absent. It reads just the finding's
`title` (one `SELECT`, not the full citation/annotation read `export`'s `.md`
path does — html export needs nothing else from Python) to compute the
default `<slug>.html` path the same way the `.md` path does via the existing
`_slug()`, and resolves that (or an explicit `--out`) to an **absolute** path
*before* invoking node. This matters because the node subprocess runs with
`cwd=serve_dir` (so Vite's `createServer({ root: serve_dir })` finds
`vite.config.js`/`svelte.config.js` without extra plumbing) — if the output
path stayed relative, "no `--out`" would write into `~/.bartleby/serve`
instead of wherever the `.md` export would have landed. `BARTLEBY_PROJECT` is
exported into the subprocess's environment exactly as `serve --project` does,
so `db.js`'s `activeProject()` picks the same corpus with no new mechanism.
The node script's own job is minimal: load the module, call
`buildFindingExportHtml(findingId)`, and `fs.writeFileSync` the result at the
absolute path it's given — no stdout protocol, no title/slug logic
duplicated on the node side.

**`FONTS_DIR` no longer depends on `process.cwd()`.** It used to be
`path.join(process.cwd(), 'static', 'fonts')`, safe only because `serve`
`os.chdir()`s into the synced tree before running vite. The CLI's node
subprocess also runs with `cwd=serve_dir` (see above) so the old line would
have kept working by coincidence — but the issue asked for cwd-independence
outright, and relying on a coincidence two call sites now share is exactly
the kind of thing that breaks quietly the next time either one changes. It
now derives from `import.meta.url` — `path.dirname(fileURLToPath(import.meta
.url))` plus three `..` — since `exportHtml.js` always sits at
`src/lib/server/exportHtml.js`, a fixed depth under the synced tree's root,
regardless of cwd or which consumer (`vite dev`'s dev server, or this script's
`ssrLoadModule`) loaded it. Both consumers resolve through the same Vite
instance and config (including `preserveSymlinks: true`), so a source
checkout's symlinked `src/` resolves identically either way.

**Verification.** A throwaway sandbox (`BARTLEBY_HOME` pointed at a temp
dir) was built directly through the typed chunk-insert helpers — a real PDF
fixture from `benchmarks/corpus/`, a generated JPEG registered as an
extracted image chunk, and a finding citing both via `[^chunk:N]` — then
`bartleby finding export finding:1 --html` was run against it both with and
without `--out`. The written file carried all six `@font-face` entries as
base64 `data:font/woff2`, one embedded `data:application/pdf`, one embedded
`data:image/jpeg`, and the finding's title/body; the no-`--out` run wrote
`<slug>.html` into the invoking shell's cwd, not `~/.bartleby/serve`. A
nonexistent `finding:<N>` and a `node`/`npm`-stripped `PATH` were also
exercised, exiting 1 with the expected one-line messages. `uv run pytest`
(untouched — no Python test added, per the no-tests-for-features rule) and
the full suite both pass.
