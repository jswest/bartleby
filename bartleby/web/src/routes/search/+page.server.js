import { runSkill, toSkillError } from '$lib/server/skill.js';
import { listTags, enrichHits, enrichScanMatches } from '$lib/server/queries.js';
import { clampInt } from '$lib/server/params.js';
import { ALL_KINDS, DEFAULT_KINDS } from '$lib/constants.js';

// The web defaults to everything a human typically wants — documents, their own
// findings, and images — but summaries stay opt-in (derived, noisier as search
// targets). ALL_KINDS is the full set the `search` script understands.

export async function load({ url }) {
  const sp = url.searchParams;
  const q = (sp.get('q') ?? '').trim();
  const mode = sp.get('mode') === 'scan' ? 'scan' : 'search';
  const kinds = sp.getAll('kind').filter((k) => ALL_KINDS.includes(k));
  const tags = sp.getAll('tag');
  const docs = (sp.get('docs') ?? '').trim();
  const context = clampInt(sp.get('context'), 0, 5, 0);
  const matchTerms = sp.get('terms') === '1';
  const offset = clampInt(sp.get('offset'), 0, Number.MAX_SAFE_INTEGER, 0);
  const limit = clampInt(sp.get('limit'), 1, 500, mode === 'scan' ? 100 : 20);

  // Echoed back so the form can re-render its current state.
  const params = { q, mode, kinds, tags, docs, context, matchTerms, offset, limit };
  const available = { tags: listTags() };

  if (!q) return { params, available, result: null, error: null };

  try {
    const args = [q, '--limit', limit];
    if (docs) args.push('--in-documents', docs);
    for (const t of tags) args.push('--tag', t);

    let result;
    if (mode === 'scan') {
      if (matchTerms) args.push('--match-terms');
      if (offset) args.push('--offset', offset);
      result = await runSkill('scan', args);
      // Add summary title/description + a source link to each match.
      result.matches = enrichScanMatches(result.matches);
    } else {
      // Empty selection (incl. "user unchecked everything" — GET forms can't
      // tell that from a first visit) falls back to the web default set.
      const effective = kinds.length ? kinds : DEFAULT_KINDS;
      for (const k of effective) args.push(`--${k}`);
      if (context) args.push('--add-context', context);
      result = await runSkill('search', args);
      result.results = enrichHits(result.results);
    }
    return { params, available, result, error: null };
  } catch (e) {
    return { params, available, result: null, error: toSkillError(e) };
  }
}
