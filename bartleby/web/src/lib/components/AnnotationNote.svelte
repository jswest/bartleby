<script>
  // One annotation's content (#734), shared by the gutter (anchored notes) and
  // the below-body block (whole-finding, stale and unplaceable notes). The
  // note text is PLAIN TEXT — Svelte-escaped, never markdown/{@html} — so it
  // needs no sanitizer. Outside the gutter an anchored note has no highlight
  // beside it, so it quotes its span; "text has changed" = the quote no
  // longer matches the body.
  import { createEventDispatcher } from "svelte";

  export let annotation;
  export let gutter = false;

  const dispatch = createEventDispatcher();
  $: stale = annotation.anchor_exact != null && !annotation.anchor_found;
</script>

<p class="margin-note__head">
  <span class="annotation-glyph" aria-hidden="true">✎</span>{annotation.is_human_author ? "human" : "agent"} note
  {#if stale}<span class="annotation-stale" title="The finding was edited after this note; its quote no longer matches.">text has changed</span>{/if}
</p>
{#if annotation.anchor_exact != null && !gutter}<p class="annotation-quote">“{annotation.anchor_exact}”</p>{/if}
<p class="annotation-body">{annotation.body}</p>
<p class="annotation-meta">
  <span title={annotation.created_at}>{annotation.created_at.slice(0, 10)}</span>
  {#if annotation.chunk_id != null}· <a href="/chunks/{annotation.chunk_id}">chunk:{annotation.chunk_id}</a>{/if}
  · <button type="button" class="annotation-delete" on:click={() => dispatch("delete", annotation)}>delete</button>
</p>
