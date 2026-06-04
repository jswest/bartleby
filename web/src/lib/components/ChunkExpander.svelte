<script>
  // Lazily fetches a chunk's full text via the read_chunks skill (through
  // /api/chunks) and shows it inline. Snippets in results are truncated; this
  // is the "read the whole passage" affordance the issue calls for.
  export let chunkId;

  let state = "idle"; // idle | loading | open | error
  let text = "";
  let error = "";

  async function toggle() {
    if (state === "open") {
      state = "idle";
      return;
    }
    if (text) {
      state = "open";
      return;
    }
    state = "loading";
    try {
      const res = await fetch(`/api/chunks?ids=${chunkId}`);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      const chunk = data.chunks?.[0];
      if (!chunk) throw new Error("chunk not found");
      text = chunk.text;
      state = "open";
    } catch (e) {
      error = e.message;
      state = "error";
    }
  }
</script>

<div class="expander">
  <button type="button" class="toggle" on:click={toggle} disabled={state === "loading"}>
    {#if state === "loading"}Loading…
    {:else if state === "open"}Hide full text
    {:else}Show full text{/if}
  </button>

  {#if state === "open"}
    <pre class="full">{text}</pre>
  {:else if state === "error"}
    <p class="err">Couldn’t load chunk: {error}</p>
  {/if}
</div>

<style>
  .expander {
    margin-top: 0.5rem;
  }
  .toggle {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    padding: 0.15rem 0.55rem;
    background: transparent;
    border: 1px solid var(--color-off);
    border-radius: 3px;
    color: var(--color-off);
    cursor: pointer;
  }
  .toggle:hover:not(:disabled) {
    background: var(--color-off);
    color: #fff;
  }
  .toggle:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .full {
    margin-top: 0.5rem;
    padding: 0.75rem;
    background: var(--color-bg-soft);
    border: 1px solid var(--color-rule);
    border-radius: 4px;
    font-family: var(--font-serif);
    font-size: 0.9rem;
    line-height: 1.5;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .err {
    margin-top: 0.4rem;
    color: #a33;
    font-size: 0.85rem;
  }
</style>
