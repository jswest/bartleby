<script>
  import { marked } from "marked";
  import { pluralize, stripExt } from "$lib/format.js";

  // One card for both search hits and scan matches — they share ~90% of their
  // shape. `variant` picks the two differences: a search hit shows a normalized
  // score badge; a scan match shows a character count and a truncation ellipsis.
  //
  // `item` is enriched server-side with title (summary/finding title),
  // description, file_name, href (document detail page at the cited page, or
  // finding page), page_number, section_heading, content_type, chunk_id, text.
  // Search adds normalized_score + source_kind + source_id; scan adds text_length.
  export let item;
  export let variant = "search"; // "search" | "scan"

  $: title = item.title ?? stripExt(item.file_name) ?? item.source_name;
  $: scorePct = Math.round((item.normalized_score ?? 0) * 100);
  $: truncated = variant === "scan" && item.text_length > item.text.length;

  // Render the snippet as markdown only when its text is reliably
  // markdown-authored: agent/model output (findings, summaries), a Docling pipe
  // table (sec_table), or a Markdown source file. Extracted PDF / plain text
  // stays literal — running it through marked would mangle stray *, _, # chars.
  // marked.parse is sanitized globally (the +layout.svelte marked.use hook wires
  // DOMPurify), so {@html} here is safe.
  $: isMarkdown =
    item.source_kind === "finding" ||
    item.source_kind === "summary" ||
    item.content_type === "sec_table" ||
    (item.file_name?.toLowerCase().endsWith(".md") ?? false);
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
      · chunk {item.chunk_id}
    {:else}
      {#if item.section_heading}{item.section_heading} · {/if}
      {#if item.content_type}{item.content_type} · {/if}
      chunk {item.chunk_id} · {pluralize(item.text_length, "char")}
    {/if}
  </p>

  {#if item.description}
    <p class="desc">{item.description}</p>
  {/if}

  {#if isMarkdown}
    <div class="snippet markdown-body markdown-body--compact">{@html marked.parse(item.text)}</div>
  {:else}
    <p class="snippet">{item.text}{#if truncated}<span class="trunc">…</span>{/if}</p>
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
  .snippet {
    margin-top: var(--space-sm);
    font-family: var(--font-serif);
    line-height: 1.5;
  }
  .trunc {
    color: var(--color-off);
  }
</style>
