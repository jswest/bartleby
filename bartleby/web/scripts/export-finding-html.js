// CLI entry point for `bartleby finding export --html` (GH-0752): writes the
// same file the "Save as HTML" button produces (GH-0690) by calling the same
// `buildFindingExportHtml`, not a Python copy of it.
//
// The builder's graph uses SvelteKit's `$lib` alias, which plain Node can't
// resolve, so the module loads through Vite's SSR loader. The config is inline
// (`configFile: false`): no `sveltekit()` plugin, so nothing is written to the
// `.svelte-kit/` a running `bartleby serve` watches, and no HMR websocket, so
// no port is bound. Everything resolves from this file's own location, so cwd
// doesn't matter.
//
// Usage: node export-finding-html.js <findingId> <absoluteTarget>
// A target that is an existing directory gets the button's `<slug>.html` file
// name; anything else is the file path. Prints the written path on stdout.
// Expects BARTLEBY_PROJECT (and optionally BARTLEBY_HOME) in the environment,
// same as `bartleby serve --project`.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

async function main() {
  const [findingIdArg, target] = process.argv.slice(2);
  const findingId = Number(findingIdArg);
  if (!Number.isInteger(findingId) || !target) {
    console.error('Usage: export-finding-html.js <findingId> <absoluteTarget>');
    process.exitCode = 1;
    return;
  }

  const server = await createServer({
    configFile: false,
    root,
    logLevel: 'error',
    resolve: { alias: { $lib: path.join(root, 'src', 'lib') }, preserveSymlinks: true },
    ssr: { external: ['better-sqlite3'] },
    server: { middlewareMode: true, hmr: false, ws: false },
    optimizeDeps: { noDiscovery: true },
    appType: 'custom'
  });
  try {
    const { buildFindingExportHtml } = await server.ssrLoadModule('/src/lib/server/exportHtml.js');
    const built = buildFindingExportHtml(findingId);
    if (!built) {
      console.error(`No finding with id ${findingId} in this corpus.`);
      process.exitCode = 1;
      return;
    }
    const isDir = fs.existsSync(target) && fs.statSync(target).isDirectory();
    const outPath = isDir ? path.join(target, built.filename) : target;
    fs.writeFileSync(outPath, built.html, 'utf-8');
    console.log(outPath);
  } finally {
    await server.close();
  }
}

main().catch((err) => {
  console.error(err?.message ?? String(err));
  process.exitCode = 1;
});
