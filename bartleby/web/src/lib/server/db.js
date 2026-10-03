import Database from 'better-sqlite3';
import yaml from 'js-yaml';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Honors BARTLEBY_HOME (set to an absolute path) like the Python side. `serve`
// is one long-lived process whose env is fixed at launch, so reading it once
// here is fine — no per-call resolution needed. See GH-0393.
const BARTLEBY_DIR = process.env.BARTLEBY_HOME || path.join(os.homedir(), '.bartleby');
const CONFIG_PATH = path.join(BARTLEBY_DIR, 'config.yaml');

function activeProject() {
  // `bartleby serve --project <name>` exports this to override the persisted
  // active project for this server only (see commands/serve.py). It wins over
  // config.yaml so both the direct DB open below and the skill subprocesses
  // (skill.js derives --project from getDb()) follow the same project.
  if (process.env.BARTLEBY_PROJECT) {
    return process.env.BARTLEBY_PROJECT;
  }
  if (!fs.existsSync(CONFIG_PATH)) {
    throw new Error(`No Bartleby config at ${CONFIG_PATH}.`);
  }
  const cfg = yaml.load(fs.readFileSync(CONFIG_PATH, 'utf8')) || {};
  if (!cfg.active_project) {
    throw new Error('No active_project in ~/.bartleby/config.yaml.');
  }
  return cfg.active_project;
}

// One memoized handle per mode — each reopens on project switch since the dev
// server outlives a single project.
function memoizedOpen(readonly) {
  let db = null;
  let openProject = null;
  return function () {
    const project = activeProject();
    if (db && openProject === project) return { db, project };

    if (db) db.close();
    const dbPath = path.join(BARTLEBY_DIR, 'projects', project, 'bartleby.db');
    if (!fs.existsSync(dbPath)) {
      throw new Error(`Project '${project}' has no database at ${dbPath}.`);
    }
    db = new Database(dbPath, { readonly, fileMustExist: true });
    if (!readonly) {
      // A skill script or CLI command may hold the write lock; wait, don't fail.
      db.pragma('busy_timeout = 5000');
      db.pragma('foreign_keys = ON');
    }
    openProject = project;
    return { db, project };
  };
}

// The read handle every page and query uses.
export const getDb = memoizedOpen(true);

// serve's ONLY writable handle (GH-0689). Imported solely by
// $lib/server/annotations.js, whose insert/delete back the annotation
// endpoints — nothing else in the web app may write. Opened lazily on first
// write, so a read-only browse never takes a writable connection.
export const getWritableDb = memoizedOpen(false);
