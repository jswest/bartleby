// CLI entry point for `bartleby finding export --html` (GH-0752): writes the
// same file the "Save as HTML" button produces (GH-0690), without a second
// copy of the builder in Python. `buildFindingExportHtml` lives in
// `$lib/server/exportHtml.js`, which is reached from deeper in its own import
// graph (`queries.js`'s `$lib/format.js`) via SvelteKit's `$lib` alias — an
// alias only Vite resolves, not plain Node `import`. Loading the module
// through Vite's programmatic SSR API (`createServer` + `ssrLoadModule`)
// resolves it exactly the way `vite dev` (what `bartleby serve` runs) does,
// so this is the same module graph, not a reimplementation. `vite` is already
// a devDependency the one-time `npm install` in `_sync_web` installs — no
// second install.
//
// Usage: node export-finding-html.js <findingId> <absoluteOutPath>
// Expects BARTLEBY_PROJECT (and optionally BARTLEBY_HOME) set in the
// environment, same as `bartleby serve --project`.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

async function main() {
  const [findingIdArg, outPathArg] = process.argv.slice(2);
  const findingId = Number(findingIdArg);
  if (!Number.isInteger(findingId) || !outPathArg) {
    console.error('Usage: export-finding-html.js <findingId> <absoluteOutPath>');
    process.exitCode = 1;
    return;
  }

  const server = await createServer({
    root,
    logLevel: 'error',
    server: { middlewareMode: true, hmr: false },
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
    fs.writeFileSync(outPathArg, built.html, 'utf-8');
  } finally {
    await server.close();
  }
}

main().catch((err) => {
  console.error(err?.message ?? String(err));
  process.exitCode = 1;
});
