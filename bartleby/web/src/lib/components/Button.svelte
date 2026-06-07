<script>
  // The one button style — token fill, off-green on hover. Used for form
  // submits (size "md") and inline toolbar actions (size "sm"). `on:click`
  // is forwarded, so callers wire handlers as usual.
  export let type = "button";
  export let size = "md"; // "md" | "sm"
</script>

<button {type} class="btn {size}" on:click>
  <slot />
</button>

<style>
  .btn {
    font-family: var(--font-sans);
    font-weight: 600;
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    color: var(--color-off);
    border-radius: var(--radius-md);
    cursor: pointer;
  }
  .btn.md {
    padding: var(--space-sm) var(--space-xl);
    font-size: var(--text-base);
  }
  .btn.sm {
    padding: var(--space-2xs) var(--space-md);
    font-size: var(--text-xs);
    border-radius: var(--radius-sm);
  }
  .btn:hover {
    background: var(--color-off);
    color: var(--color-surface);
    border-color: var(--color-off);
  }
</style>
