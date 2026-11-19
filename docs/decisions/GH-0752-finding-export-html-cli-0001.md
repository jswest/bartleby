# `finding export --html` reaches GH-0690's builder through an inline-config Vite SSR loader that leaves a running `serve` untouched

`bartleby finding export finding:<N> --html [--out PATH]` (issue #752) writes
the same self-contained file `serve`'s "Save as HTML" button saves (GH-0690),
for an agent that only has the CLI: no `bartleby serve` and no browser. The
proposal said `buildFindingExportHtml`
(`bartleby/web/src/lib/server/exportHtml.js`) must be reused, not ported to
Python. With one builder, the button and the CLI can't drift apart.

**The entry point loads the builder through Vite with an inline config.**
`exportHtml.js`'s graph reaches `queries.js`'s
`import { clampSnippet } from '$lib/format.js'`. That is SvelteKit's `$lib`
alias, and plain Node can't resolve it. The `$lib` alias is the only thing the
graph needs from the toolchain: it has no `$app/*`, no `$env/*`, and no
`import.meta.env`. So `bartleby/web/scripts/export-finding-html.js` calls
Vite's `createServer` with `configFile: false` and only that alias,
`preserveSymlinks`, `better-sqlite3` external, `hmr: false`, `ws: false`, and
`optimizeDeps.noDiscovery`. Then it calls `ssrLoadModule`. `vite` is already a
devDependency of the one-time `npm install`, so nothing new gets installed.

The first version loaded `vite.config.js`, which runs the `sveltekit()`
plugin. That plugin rewrites `~/.bartleby/serve/.svelte-kit/generated/*`. A
live `serve` watches that directory, so every export sent a `full-reload` to
every open tab. Its default HMR websocket also tried to bind port 24678. The
inline config has no plugin, so it doesn't write into `.svelte-kit/`. With
`ws: false`, Vite uses a no-op websocket stub, so no port is bound. The script
resolves `root` and `vite` from its own file location, so the caller's cwd
doesn't matter.

**Node owns naming and the missing-finding case.** `buildFindingExportHtml`
already returns `{ html, filename }`, or `null` for a missing finding.
Python passes one absolute target: the resolved `--out`, or the caller's cwd
when `--out` is omitted. If the target is an existing directory, node writes
`<target>/<filename>`, which is the button's `<slug>.html`. Otherwise it
writes the target path itself. Node prints the written path on stdout, and
Python reports it. A missing finding makes node exit 1 with a one-line
message.

Python makes no finding lookup. It still opens the DB once, through
`_open_or_exit`, so a missing corpus or a stale schema gives the same error the
`.md` export gives. Otherwise the web code would open the DB raw and fail
later with a less clear error. `BARTLEBY_PROJECT` is set in the subprocess
environment, exactly as `serve --project` does.

**One `prepare_web()`, safe to run concurrently and under a live `serve`.**
`commands/serve.py`'s `prepare_web() -> Path` syncs the UI into
`~/.bartleby/serve`, runs `npm install` if needed, and returns that
directory. Both `serve.main` and the export call it, and `finding.py` imports
only `prepare_web` and `_require_node`. Two changes make it safe:

- `_sync_web` now leaves alone any entry that already matches its source. A
  matching entry is either the `src` symlink pointing at the same target, or
  a file or tree whose (path, size, mtime) signature matches, since `copy2`
  preserves mtimes. Only stale entries are removed and re-created. In the
  steady state, an export under a running `serve` writes nothing into that
  tree. Before this change, the export rewrote `vite.config.js` and the `src`
  symlink, and vite saw both changes.
- An exclusive `fcntl.flock` on `~/.bartleby/serve.lock` wraps the sync and
  the install. The lock file sits beside the tree, not inside it, so the sync
  can't delete it. Two exports, or an export and a `serve` start, can no
  longer delete and copy over each other. In testing, 2 of 9 parallel runs
  hit `FileNotFoundError` without the lock.

**`FONTS_DIR` doesn't depend on `process.cwd()`.** It used to be
`process.cwd()/static/fonts`. That only worked because `serve` changes into
the synced tree before it runs vite. It is now located from
`import.meta.url`: `exportHtml.js` always sits at `src/lib/server/` under the
synced root, so the fonts are three directories up. Because
`preserveSymlinks` is set, a source checkout's symlinked `src/` resolves to
`~/.bartleby/serve/static/fonts` in both consumers.

**Verification.** All checks ran in a throwaway `BARTLEBY_HOME` with a
seeded finding.

- **Running the export.** It was run from a cwd other than the serve
  directory, both with no `--out` and with `--out` set to a file or a
  directory. It wrote the button's file name, with all six `@font-face`
  `data:` URIs. A missing `finding:99` exited 1 with
  `No finding with id 99 in this corpus.` A missing project gave the
  `open_db` error.
- **Parallel runs.** Nine parallel exports succeeded, both with the tree
  already in sync and with the synced entries deleted first. No run printed a
  traceback.
- **Under a live `serve`.** A throwaway `vite dev` ran on a spare port, with
  headless Chrome on `/findings/1`, a marker set in the page through CDP, and
  a separate HMR websocket listener. One export and then five parallel exports
  sent no `full-reload` frames and logged no `page reload` lines, and the
  in-page marker survived. Running the old entry point against the same
  server wiped the marker and sent seven `full-reload` frames from
  `.svelte-kit/generated/*`.
- **Busy HMR port.** With port 24678 held by another process, the export
  printed no websocket or port error.
