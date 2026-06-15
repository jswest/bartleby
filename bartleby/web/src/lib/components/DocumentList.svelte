<script>
  import { pluralize, stripExt } from "$lib/format.js";
  // Shared document listing — used by /documents and /tags/[id]. Each document
  // carries a `tags` array (possibly empty); untagged docs render no chips.
  export let documents;
</script>

<ul class="card-grid">
  {#each documents as d}
    <li class="entity surface surface--interactive">
      <h2>
        <a href="/documents/{d.document_id}">
          {d.title ?? stripExt(d.file_name)}
        </a>
      </h2>
      <p class="ident">{d.file_name}</p>
      {#if d.page_count || !d.title}
        <p class="meta">
          {#if d.page_count}{pluralize(d.page_count, "page")}{/if}
          {#if !d.title}{#if d.page_count} · {/if}<em>no summary</em>{/if}
        </p>
      {/if}
      {#if d.tags.length}
        <div class="tags">
          {#each d.tags as t}
            <span class="tag" title={t.description}>{t.name}</span>
          {/each}
        </div>
      {/if}
      {#if d.description}
        <p class="desc">{d.description}</p>
      {/if}
    </li>
  {/each}
</ul>
