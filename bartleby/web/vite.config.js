import { sveltekit } from '@sveltejs/kit/vite';

export default {
  plugins: [sveltekit()],
  // better-sqlite3 is a native CommonJS module; don't try to pre-bundle it.
  ssr: { external: ['better-sqlite3'] },
  // From a source checkout, `bartleby serve` symlinks ~/.bartleby/serve/src →
  // the repo's bartleby/web/src so edits hot-reload. Without preserveSymlinks,
  // Node resolves the realpath and then can't find node_modules (which lives
  // under ~/.bartleby/serve, not the repo). Treat the symlink as if it were the
  // directory it appears to be. (Installed packages copy src/ instead, so this
  // is a no-op there.)
  resolve: { preserveSymlinks: true }
};
