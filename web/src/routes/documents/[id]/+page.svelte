<script>
  import { marked } from "marked";
  import { page } from "$app/stores";
  import { stripExt } from "$lib/format.js";
  export let data;

  $: doc = data.document;
  $: summaryHtml = doc.summary_text ? marked.parse(doc.summary_text) : null;

  // ?page=N (set by search/scan result links) jumps the PDF viewer to the
  // cited page. Sanitized to a positive integer; anything else opens page 1.
  $: pageNum = (() => {
    const n = parseInt($page.url.searchParams.get("page"), 10);
    return Number.isInteger(n) && n > 0 ? n : null;
  })();
  $: viewerSrc = `/files/${doc.document_id}${pageNum ? `#page=${pageNum}` : ""}`;
</script>

<div class="split">
  <article class="report">
    <h1>{doc.title ?? stripExt(doc.file_name)}</h1>
    <p class="meta">
      {doc.file_name}{doc.page_count ? ` · ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}` : ""}
      {#if doc.model} · summarized by {doc.model}{/if}
    </p>
    {#if doc.description}
      <p class="desc">{doc.description}</p>
    {/if}

    {#if doc.tags.length}
      <div class="tags">
        {#each doc.tags as t}
          <span class="tag" title={t.description}>{t.name}</span>
        {/each}
      </div>
    {/if}

    {#if summaryHtml}
      <div class="body markdown-body">
        {@html summaryHtml}
      </div>
    {:else}
      <p class="empty">No summary stored for this document.</p>
    {/if}
  </article>

  <aside class="viewer">
    {#key viewerSrc}
      <iframe title="Source document" src={viewerSrc}></iframe>
    {/key}
  </aside>
</div>

<style>
  .empty {
    margin-top: 2rem;
  }
</style>
