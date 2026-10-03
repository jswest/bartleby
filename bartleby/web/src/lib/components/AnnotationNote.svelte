<script>
  // One annotation's content (#734), shared by the gutter (anchored notes) and
  // the below-body block (whole-finding and stale notes). The note text is
  // PLAIN TEXT — Svelte-escaped, never markdown/{@html} — so it needs no
  // sanitizer. `stale` = anchored, but the quote no longer matches the body.
  import { createEventDispatcher } from "svelte";

  export let annotation;
  export let stale = false;

  const dispatch = createEventDispatcher();
  $: a = annotation;
</script>

<p class="margin-note__head">
  <span class="annotation-glyph" aria-hidden="true">✎</span>{a.is_human_author ? "human" : "agent"} note
  {#if stale}<span class="annotation-stale" title="The finding was edited after this note; its quote no longer matches.">text has changed</span>{/if}
</p>
{#if stale}<p class="annotation-quote">“{a.anchor_exact}”</p>{/if}
<p class="annotation-body">{a.body}</p>
<p class="annotation-meta">
  <span title={a.created_at}>{a.created_at.slice(0, 10)}</span>
  {#if a.chunk_id != null}· <a href="/chunks/{a.chunk_id}">chunk:{a.chunk_id}</a>{/if}
  · <button type="button" class="annotation-delete" on:click={() => dispatch("delete", a)}>delete</button>
</p>
