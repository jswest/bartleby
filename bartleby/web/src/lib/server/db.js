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

let _db = null;
let _project = null;

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

// Memoized — reopens on project switch since the dev server outlives a single project.
export function getDb() {
  const project = activeProject();
  if (_db && _project === project) return { db: _db, project };

  if (_db) _db.close();
  const dbPath = path.join(BARTLEBY_DIR, 'projects', project, 'bartleby.db');
  if (!fs.existsSync(dbPath)) {
    throw new Error(`Project '${project}' has no database at ${dbPath}.`);
  }
  _db = new Database(dbPath, { readonly: true, fileMustExist: true });
  _project = project;
  return { db: _db, project };
}
