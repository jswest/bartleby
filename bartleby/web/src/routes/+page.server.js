import { runSkill, toSkillError } from '$lib/server/skill.js';
import { getCounts, listAllTags } from '$lib/server/queries.js';

// The home page is a corpus overview: nav-card counts (findings live outside
// describe_corpus, which is document-centric) plus the full describe_corpus
// aggregate. We delegate to the skill rather than re-deriving its SQL — it's
// pure, sub-millisecond, and keeps one source of truth for "what is this
// corpus?". The skill omits tag_id (agents address tags by name); the /tags
// pages key on id, so we splice it back in from listAllTags().
export async function load() {
  const counts = getCounts();
  try {
    const corpus = await runSkill('describe_corpus', ['--top-n', 8]);
    const tagIds = new Map(listAllTags().map((t) => [t.name, t.tag_id]));
    corpus.tags = corpus.tags.map((t) => ({ ...t, tag_id: tagIds.get(t.name) ?? null }));
    return { counts, corpus, error: null };
  } catch (e) {
    // A failed overview shouldn't blank the page — fall back to the nav cards.
    return { counts, corpus: null, error: toSkillError(e) };
  }
}
