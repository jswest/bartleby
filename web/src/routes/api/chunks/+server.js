import { json, error } from '@sveltejs/kit';
import { runSkill, SkillError } from '$lib/server/skill.js';

// Full chunk text by id, via the read_chunks skill. Backs the inline
// "Show full text" expander on search/scan results (snippets are truncated).
export async function GET({ url }) {
  const raw = url.searchParams.get('ids') ?? '';
  // Accept a comma-separated list; keep only positive integers.
  const ids = raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => /^\d+$/.test(s));

  if (ids.length === 0) throw error(400, 'no valid chunk ids');

  try {
    const result = await runSkill('read_chunks', ['--chunks', ids.join(',')]);
    return json({ chunks: result.chunks, missing: result.missing ?? [] });
  } catch (e) {
    const code = e instanceof SkillError ? e.code : 'ERROR';
    return json({ error: e.message, code }, { status: 502 });
  }
}
