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
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
  }
  .back a {
    color: var(--color-off);
    text-decoration: none;
  }
  .tag-desc {
    max-width: 44rem;
    font-style: italic;
    color: #444;
    margin-bottom: 1.5rem;
  }
</style>
