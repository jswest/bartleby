<script>
  import DocumentList from "$lib/components/DocumentList.svelte";
  export let data;
</script>

<h1>Documents</h1>

{#if data.documents.length === 0}
  <p class="empty">No documents ingested yet.</p>
{:else}
  <DocumentList documents={data.documents} />
{/if}
