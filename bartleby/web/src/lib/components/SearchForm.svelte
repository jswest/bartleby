<script>
  import { ALL_KINDS, DEFAULT_KINDS } from "$lib/constants.js";
  import Button from "$lib/components/Button.svelte";

  export let params;
  export let availableTags = [];

  // Local copy of the mode so the form can show/hide mode-specific fields
  // before submit. Everything else submits straight from the inputs.
  let mode = params.mode;

  $: kindChecked = (k) =>
    params.kinds.length ? params.kinds.includes(k) : DEFAULT_KINDS.includes(k);
  $: tagChecked = (name) => params.tags.includes(name);
</script>

<form class="search-form" method="GET" action="/search">
  <div class="mode" role="radiogroup" aria-label="Search mode">
    <label class:active={mode === "search"}>
      <input type="radio" name="mode" value="search" bind:group={mode} />
      Search <span class="mode-hint">ranked</span>
    </label>
    <label class:active={mode === "scan"}>
      <input type="radio" name="mode" value="scan" bind:group={mode} />
      Scan <span class="mode-hint">enumerate</span>
    </label>
  </div>

  <div class="query-row">
    <input
      class="query"
      type="search"
      name="q"
      value={params.q}
      placeholder={mode === "scan"
        ? "Phrase to enumerate across documents…"
        : "Search documents, findings, images…"}
      autocomplete="off"
      autofocus
    />
    <Button type="submit">{mode === "scan" ? "Scan" : "Search"}</Button>
  </div>

  <div class="filters">
    {#if mode === "search"}
      <fieldset>
        <legend>Sources</legend>
        <div class="checks">
          {#each ALL_KINDS as k}
            <label class="check">
              <input type="checkbox" name="kind" value={k} checked={kindChecked(k)} />
              {k}
            </label>
          {/each}
        </div>
      </fieldset>

      <fieldset>
        <legend>Context</legend>
        <label class="inline">
          <input type="number" name="context" min="0" max="5" value={params.context} />
          neighbor chunks
        </label>
      </fieldset>
    {:else}
      <fieldset>
        <legend>Matching</legend>
        <label class="check">
          <input type="checkbox" name="terms" value="1" checked={params.matchTerms} />
          terms (AND) instead of exact phrase
        </label>
      </fieldset>
    {/if}

    {#if availableTags.length}
      <fieldset>
        <legend>Tags</legend>
        <div class="checks">
          {#each availableTags as t}
            <label class="check">
              <input type="checkbox" name="tag" value={t.name} checked={tagChecked(t.name)} />
              {t.name} <span class="tag-count">{t.document_count}</span>
            </label>
          {/each}
        </div>
      </fieldset>
    {/if}

    <fieldset>
      <legend>Scope</legend>
      <label class="inline">
        <input
          class="docs"
          type="text"
          name="docs"
          value={params.docs}
          placeholder="document ids, e.g. 3,5,9"
        />
      </label>
      <label class="inline">
        limit
        <input type="number" name="limit" min="1" max="500" value={params.limit} />
      </label>
    </fieldset>
  </div>
</form>

<style>
  /* The query box, mode toggle, and field widths are search-form-specific;
     the fieldset/legend/checks/inputs chrome lives in app.css (shared with the
     /documents filter form). */
  .search-form {
    margin-bottom: var(--space-xl);
    font-family: var(--font-sans);
  }
  .mode {
    display: inline-flex;
    border: 1px solid var(--color-off);
    border-radius: var(--radius-md);
    overflow: hidden;
    margin-bottom: var(--space-md);
  }
  .mode label {
    padding: var(--space-xs) var(--space-md);
    cursor: pointer;
    color: var(--color-off);
    background: var(--color-surface);
    font-size: var(--text-sm);
  }
  .mode label.active {
    background: var(--color-off);
    color: var(--color-off-light);
  }
  .mode input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }
  .mode-hint {
    opacity: 0.6;
    font-size: var(--text-2xs);
  }
  .query-row {
    display: flex;
    gap: var(--space-sm);
    max-width: var(--width-content);
  }
  /* The query box is a paper field on the dark shell: white ground, dark ink,
     Iosevka chrome (inputs are UI, not prose — so sans, not serif). */
  .query {
    flex: 1;
    padding: var(--space-sm) var(--space-md);
    font-size: var(--text-base);
    font-family: var(--font-sans);
    background: var(--color-surface);
    color: var(--color-text);
    border: 1px solid var(--color-off);
    border-radius: var(--radius-md);
  }
  .docs {
    width: 12rem;
  }
</style>
