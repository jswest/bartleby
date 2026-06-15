<script>
  import { marked } from "marked";
  import { pluralize, stripExt, isMarkdownChunk, highlightTerms } from "$lib/format.js";
  import { CHUNK_ICON as chunkIcon } from "$lib/icons.js";

  // One card for both search hits and scan matches — they share ~90% of their
  // shape. `variant` picks the two differences: a search hit shows a normalized
  // score badge; a scan match shows a character count.
  //
  // `item` is enriched server-side with title (summary/finding title),
  // description, file_name, href (document detail page at the cited page, or
  // finding page), page_number, section_heading, content_type, chunk_id, text.
  // Search adds normalized_score + source_kind + source_id; scan adds text_length.
  // `query` (the raw query string) is passed through so matched terms can be
  // wrapped in <mark> on the client — no raw HTML injection, just Svelte {#each}.
  export let item;
  export let variant = "search"; // "search" | "scan"
  export let query = "";

  $: title = item.title ?? stripExt(item.file_name) ?? item.source_name;
  $: scorePct = Math.round((item.normalized_score ?? 0) * 100);

  // The "open chunk in context" link → /chunks/<id>. Built as a string (chunk_id
  // is an integer, so no escaping) and {@html}'d so the identical markup isn't
  // duplicated across the two meta-line variants. CHUNK_ICON is inline SVG.
  $: chunkLink =
    `<a class="chunk-link" href="/chunks/${item.chunk_id}"` +
    ` title="Open chunk ${item.chunk_id} in context"` +
    ` aria-label="Open chunk ${item.chunk_id} in context">${chunkIcon}</a>`;

  // Render the snippet as markdown only when its text is reliably
  // markdown-authored (see isMarkdownChunk). Extracted PDF / plain text stays
  // literal — running it through marked would mangle stray *, _, # chars.
  // marked.parse is sanitized globally (the +layout.svelte marked.use hook wires
  // DOMPurify), so {@html} here is safe.
  $: isMarkdown = isMarkdownChunk(item);

  // B3: split the plain-text snippet into {text, highlight} segments so the
  // template can wrap matched terms in <mark> without any raw-HTML injection.
  // Markdown snippets go through `marked` and keep their own emphasis; we leave
  // them unmarked (injecting <mark> mid-parse would corrupt the AST).
  $: snippetSegments = isMarkdown ? null : highlightTerms(item.text, query);
</script>

<li class="result surface surface--interactive">
  <div class="head">
    <h3 class="title">
      {#if item.href}
        <a href={item.href}>
          {title}{#if item.page_number}<span class="page-hint"> p.{item.page_number}</span>{/if}
        </a>
      {:else}
        {title}
      {/if}
    </h3>
    {#if variant === "search"}
      <span class="score" title="normalized score (top hit = 100)">{scorePct}</span>
    {/if}
  </div>

  {#if item.file_name || item.source_kind === "finding"}
    <p class="ident">
      {#if item.file_name}{item.file_name}{:else}<span class="finding-id">#{item.source_id}</span>{/if}
    </p>
  {/if}

  <p class="meta">
    {#if variant === "search"}
      <span class="kind">{item.source_kind}</span>
      {#if item.section_heading} · {item.section_heading}{/if}
      {#if item.content_type} · {item.content_type}{/if}
      · chunk {item.chunk_id}{@html chunkLink}
    {:else}
      {#if item.section_heading}{item.section_heading} · {/if}
      {#if item.content_type}{item.content_type} · {/if}
      chunk {item.chunk_id}{@html chunkLink} · {pluralize(item.text_length, "char")}
    {/if}
  </p>

  {#if item.description}
    <p class="desc">{item.description}</p>
  {/if}

  {#if isMarkdown}
    <div class="snippet markdown-body markdown-body--compact">{@html marked.parse(item.text)}</div>
  {:else}
    <p class="snippet">
      {#each snippetSegments as seg}
        {#if seg.highlight}<mark>{seg.text}</mark>{:else}{seg.text}{/if}
      {/each}
    </p>
  {/if}
</li>

<style>
  .result {
    list-style: none;
    margin-bottom: var(--space-lg);
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: var(--space-sm);
  }
  .title {
    flex: 1;
    font-size: var(--text-base);
    font-weight: 600;
    min-width: 0;
  }
  .title a {
    color: inherit;
    text-decoration: none;
  }
  .title a:hover {
    color: var(--color-off);
  }
  .page-hint {
    font-family: var(--font-sans);
    font-size: var(--text-2xs);
    font-weight: 600;
    color: var(--color-token-text);
    white-space: nowrap;
  }
  .score {
    font-family: var(--font-display);
    font-size: var(--text-xs);
    color: var(--color-off);
    background: var(--color-surface);
    border: 1px solid var(--color-off);
    border-radius: var(--radius-sm);
    padding: 0 var(--space-xs);
  }
  .desc {
    font-size: var(--text-sm);
    margin-top: var(--space-2xs);
  }
  /* Inherit the global .meta type (sans, --text-xs) so the card's meta line
     matches the documents/findings lists; only the card's spacing lives here. */
  .meta {
    margin-top: var(--space-2xs);
  }
  .kind {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
  }
  /* Tiny "open chunk" affordance riding the meta line, right after the chunk id.
     Sits on the baseline of the surrounding text; brightens to the link color on
     hover so it's discoverable without shouting. */
  .chunk-link {
    display: inline-flex;
    vertical-align: baseline;
    margin-left: var(--space-3xs);
    color: var(--color-off);
  }
  .chunk-link:hover {
    color: var(--color-link);
  }
  /* The glyph is injected via {@html}, so it sits outside Svelte's scoping —
     reach it with :global (the .chunk-link ancestor is still scoped). */
  .chunk-link :global(.icon) {
    position: relative;
    top: 1px;
  }
  .snippet {
    margin-top: var(--space-sm);
    font-family: var(--font-serif);
    line-height: 1.5;
  }
</style>
