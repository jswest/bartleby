// Pure citation-marker → dagger-note parsing, shared by the live finding view
// (routes/findings/[id]/+page.svelte) and the standalone HTML export
// (routes/findings/[id]/export.html/+server.js — GH-0690). Both need the
// *same* `[^scheme:ref]` substitution for the export's "looks exactly like
// the live page" promise to hold; only the final render shell differs (a
// Svelte reactive template vs. a hand-built static document), so this module
// owns just the parsing/note-building and leaves rendering (marked.parse,
// the surrounding HTML/DOM shape) to each caller.
//
// One type-tagged marker grammar (issue #624): [^<scheme>:<ref>].
//   [^chunk:N]         corpus-chunk citation. Resolves against byId →
//                        resolved  → ¶ "source" note (amber, filename, jumps)
//                        unresolved → ‡ "no longer available" note (danger)
//   [^finding:N]       finding-to-finding link (#654) → § note with a link
//                        to /findings/N. No DB row — validated at save time,
//                        rendered from body text on read.
//   [^url:…]/[^doc:…]   external citation → † "source" note (link / ref)
// Glyph mapping (supersedes GH-0654): ¶ corpus chunk (resolved), ‡ chunk
// gone, § finding-to-finding link, † external url/doc — the glyph alone
// tells the kind apart; the ordinal stays the tie-back.
// Mirrors the backend grammar in skill_scripts/_common.py: `chunk` and
// `finding` are internal schemes; `url`/`doc` are external; any other scheme
// (incl. `document`) is dropped, not rendered as a citation.
import { escapeHtml, stripExt } from "./format.js";

// Replace every `[^scheme:ref]` marker in `body` with an inline
// `<sup class="cite-ref ...">` anchor and collect the matching gutter note,
// in one left-to-right pass (so gutter order matches reading order).
// Returns `{ markdown, notes }` — `markdown` still needs a `marked.parse()`
// pass (the caller's own marked instance + sanitizer); `notes` is the ordered
// gutter-note data, rendered into markup by the caller.
export function substituteCitations(body, byId, active = null) {
  const collected = [];
  let n = 0;
  const markdown = body.replace(/\[\^([A-Za-z]+):([^\]]+)\]/g, (match, scheme, ref) => {
    const s = scheme.toLowerCase();
    if (s === "chunk") {
      const id = Number(ref.trim());
      // A non-numeric chunk ref can't resolve; leave the marker verbatim.
      if (!Number.isInteger(id)) return escapeHtml(match);
      const c = byId.get(id);
      const note = c
        ? sourceNote(++n, c, c === active)
        : goneNote(++n, id);
      collected.push(note);
      return marker(note);
    }
    if (s === "finding") {
      const id = Number(ref.trim());
      // A non-numeric finding ref can't resolve; leave the marker verbatim.
      if (!Number.isInteger(id)) return escapeHtml(match);
      const note = findingNote(++n, id);
      collected.push(note);
      return marker(note);
    }
    const note = externalNote(++n, s, ref.trim());
    if (!note) {
      n--; // unknown scheme: drop, don't burn an ordinal
      return escapeHtml(match);
    }
    collected.push(note);
    return marker(note);
  });
  return { markdown, notes: collected };
}

// The inline dagger anchor that sits at the cited point in the prose. A
// superscript <sup> with the dagger glyph + ordinal; the ordinal ties it to
// its gutter note. Stable class hooks (`cite-ref`, kind modifier) so R3 (#592)
// can splice a pixel icon in without touching this code.
function marker(note) {
  const kindCls = note.gone
    ? "cite-ref--gone"
    : note.finding
      ? "cite-ref--finding"
      : "cite-ref--source";
  const title = note.gone
    ? `${note.dagger} cited source no longer available`
    : note.finding
      ? `${note.dagger} finding · ${note.title}`
      : `${note.dagger} source · ${note.title}`;
  return `<sup class="cite-ref ${kindCls}" data-note="${note.n}" title="${escapeHtml(title)}">${note.dagger}${note.n}</sup>`;
}

function sourceNote(n, c, isActive) {
  const name = stripExt(c.file_name);
  const label =
    name && c.page_number ? `${name} p.${c.page_number}`
    : name ? name
    : c.page_number ? `p.${c.page_number}`
    : `chunk ${c.chunk_id}`;
  const title = [
    c.file_name,
    c.page_number && `page ${c.page_number}`,
    `chunk ${c.chunk_id}`,
  ].filter(Boolean).join(" · ");
  return {
    n,
    dagger: "¶",
    gone: false,
    chunkId: c.chunk_id,
    active: isActive,
    label,
    title,
  };
}

// An unresolved [^chunk:N]: the cited source has been removed (deleted, or
// rebuilt under new chunk ids by an edit/merge), so only the id survives.
// Danger-toned gutter note rather than leaking raw [^N] text.
function goneNote(n, chunkId) {
  return {
    n,
    dagger: "‡",
    gone: true,
    chunkId,
    label: "no longer available",
    title: `chunk ${chunkId} · cited source no longer available`,
  };
}

// A [^finding:N] link — a unidirectional reference to another finding (#654).
// Carries an `href` to /findings/N for the live page; the export renderer
// (GH-0690) deliberately ignores it and renders an inert note instead — the
// other finding isn't in the standalone file.
function findingNote(n, findingId) {
  return {
    n,
    dagger: "§",
    gone: false,
    finding: true,
    findingId,
    href: `/findings/${findingId}`,
    label: `finding #${findingId}`,
    title: `finding #${findingId}`,
  };
}

// An external (URL / external-dataset doc) citation parsed straight from the
// body marker — no DB row backs it. A url scheme becomes a real link, a doc
// scheme a muted ref. Unknown schemes return null (dropped upstream).
function externalNote(n, scheme, ref) {
  if (scheme === "url") {
    return { n, dagger: "†", gone: false, external: "url", href: ref, label: ref, title: `external source · ${ref}` };
  }
  if (scheme === "doc") {
    return { n, dagger: "†", gone: false, external: "doc", label: ref, title: `external dataset doc · ${ref}` };
  }
  return null;
}
