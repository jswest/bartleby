import { spawn } from 'node:child_process';
import { getDb } from './db.js';

// The dedicated, memory-enabled session every web invocation runs under. The
// Python runner (bartleby/skill_runner.py) picks this up from the environment,
// finds-or-creates the matching session, and crucially never touches the
// project's .active_session pointer — so the web never collides with whichever
// session an agent has active. See bartleby.session.ensure_named_session.
const WEB_SESSION_NAME = 'web-reader';

// The CLI entrypoint. Inherited PATH resolves it when the UI was launched via
// `bartleby serve`; BARTLEBY_BIN overrides for unusual setups.
const BARTLEBY_BIN = process.env.BARTLEBY_BIN || 'bartleby';

// How long a single skill invocation may run before we give up. Semantic
// search cold-loads the BGE model, so this is generous.
const TIMEOUT_MS = 120_000;

// An error carrying the skill's structured {error, code} envelope, so routes
// can render the same diagnostics the agent would see.
export class SkillError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'SkillError';
    this.code = code ?? 'SKILL_FAILED';
  }
}

// Normalize any thrown value into the {message, code} shape routes render. A
// SkillError carries the skill's own code; anything else is an unexpected
// failure tagged 'ERROR'. Shared by every loader that calls runSkill.
export function toSkillError(e) {
  return e instanceof SkillError
    ? { message: e.message, code: e.code }
    : { message: String(e?.message ?? e), code: 'ERROR' };
}

/**
 * Run one skill script as a subprocess and return its parsed JSON.
 *
 * This is the web's single seam onto the Python skill: search/scan/read_chunks
 * all share one ranking + filtering implementation, and we render what they
 * emit rather than reimplementing any of it in JS.
 *
 * @param {string} name  skill script name, e.g. "search"
 * @param {Array<string|number>} args  CLI args (the active --project is added)
 * @returns {Promise<object>} the script's stdout JSON
 */
export function runSkill(name, args = []) {
  const { project } = getDb();
  // --project leads the user args so a trailing `--` sentinel (search/scan use
  // one to fence a leading-dash query) keeps everything after it positional.
  const argv = ['skill', name, '--project', project, ...args.map(String)];

  return new Promise((resolve, reject) => {
    const child = spawn(BARTLEBY_BIN, argv, {
      env: { ...process.env, BARTLEBY_SESSION_NAME: WEB_SESSION_NAME }
    });

    let stdout = '';
    let stderr = '';
    let settled = false;

    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(() => reject(new SkillError(`skill ${name} timed out after ${TIMEOUT_MS}ms`, 'TIMEOUT')));
    }, TIMEOUT_MS);

    function finish(fn) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    }

    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));

    child.on('error', (err) => {
      // Most commonly ENOENT — `bartleby` not on PATH.
      finish(() => reject(new SkillError(
        `could not run ${BARTLEBY_BIN}: ${err.message}`, 'SPAWN_FAILED'
      )));
    });

    child.on('close', (exitCode) => finish(() => {
      let parsed;
      try {
        parsed = JSON.parse(stdout);
      } catch {
        const detail = stderr.trim() || stdout.trim().slice(0, 200) || '(no output)';
        return reject(new SkillError(`skill ${name} emitted non-JSON: ${detail}`, 'BAD_OUTPUT'));
      }
      // Scripts signal failure with exit 1 and an {error, code} envelope.
      if (exitCode !== 0 || parsed?.error) {
        return reject(new SkillError(parsed?.error ?? `skill ${name} exited ${exitCode}`, parsed?.code));
      }
      resolve(parsed);
    }));
  });
}
