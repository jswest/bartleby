<script>
  import DocumentList from "$lib/components/DocumentList.svelte";
  export let data;

  $: tag = data.tag;
</script>

<p class="back"><a href="/tags">← Tags</a></p>
<h1>{tag.name}</h1>
{#if tag.description}
  <p class="desc tag-desc">{tag.description}</p>
{/if}

{#if tag.documents.length === 0}
  <p class="empty">No documents carry this tag.</p>
{:else}
  <DocumentList documents={tag.documents} />
{/if}

<style>
  .back {
    font-family: var(--font-sans);
    font-size: var(--text-sm);
    margin-bottom: var(--space-sm);
  }
  .back a {
    color: var(--color-off);
    text-decoration: none;
  }
  .tag-desc {
    max-width: var(--width-content);
    font-style: italic;
    margin-bottom: var(--space-xl);
  }
</style>
