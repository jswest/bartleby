import { runSkill, toSkillError } from '$lib/server/skill.js';
import { listTags, tagsByDocument } from '$lib/server/queries.js';
import { clampInt } from '$lib/server/params.js';

const SORTS = ['title', 'date', 'id'];

// /documents delegates enumeration + filtering to the list_documents skill so
// the date/null-handling semantics live in exactly one place. The skill is
// agent-shaped (id-ordered, no tags), so we pass --sort for human-friendly
// ordering and re-attach tag chips with a single batched query.
export async function load({ url }) {
  const sp = url.searchParams;
  const tags = sp.getAll('tag');
  const after = (sp.get('after') ?? '').trim();
  const before = (sp.get('before') ?? '').trim();
  const includeNulls = sp.get('nulls') === '1';
  const sort = SORTS.includes(sp.get('sort')) ? sp.get('sort') : 'title';
  const offset = clampInt(sp.get('offset'), 0, Number.MAX_SAFE_INTEGER, 0);
  const limit = clampInt(sp.get('limit'), 1, 500, 50);

  // Echoed back so the filter form can re-render its state.
  const params = { tags, after, before, includeNulls, sort, offset, limit };
  const availableTags = listTags();

  try {
    const args = ['--verbose', '--sort', sort, '--limit', limit, '--offset', offset];
    for (const t of tags) args.push('--tag', t);
    if (after) args.push('--authored-after', after);
    if (before) args.push('--authored-before', before);
    if (includeNulls) args.push('--include-nulls');

    const result = await runSkill('list_documents', args);

    // Reshape the skill's agent rows into what DocumentList renders, and splice
    // tags back in (the skill omits them).
    const tagMap = tagsByDocument(result.documents.map((d) => d.id));
    const documents = result.documents.map((d) => ({
      document_id: d.id,
      file_name: d.file_name,
      title: d.title,
      description: d.description,
      page_count: d.page_count,
      tags: tagMap.get(d.id) ?? []
    }));

    return {
      params,
      availableTags,
      documents,
      total: result.total,
      // excluded_null_dated now rides inside the nested `filters` echo, present
      // only when a scope filter (tag/date bound) is active; 0 otherwise.
      excludedNullDated: result.filters?.excluded_null_dated ?? 0,
      error: null
    };
  } catch (e) {
    return { params, availableTags, documents: [], total: 0, excludedNullDated: 0, error: toSkillError(e) };
  }
}
