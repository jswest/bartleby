import Database from 'better-sqlite3';
import yaml from 'js-yaml';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const BARTLEBY_DIR = path.join(os.homedir(), '.bartleby');
const CONFIG_PATH = path.join(BARTLEBY_DIR, 'config.yaml');

let _db = null;
let _project = null;

function activeProject() {
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
