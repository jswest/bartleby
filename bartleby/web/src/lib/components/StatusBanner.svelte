<script>
  // A full-width status box above a result/list area. "error" is the danger
  // palette; "busy" is the in-progress token palette. Slot carries the message
  // (and any inline .code / .hint spans, styled by the caller).
  export let variant = "error"; // "error" | "busy"
</script>

<div class="banner {variant}" aria-live={variant === "busy" ? "polite" : undefined}>
  <slot />
</div>

<style>
  .banner {
    font-family: var(--font-sans);
    font-size: var(--text-sm);
    padding: var(--space-sm) var(--space-md);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-lg);
    max-width: var(--width-content);
  }
  .banner.error {
    background: var(--color-danger-bg);
    border: 1px solid var(--color-danger-border);
    color: var(--color-danger-text);
  }
  .banner.busy {
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    color: var(--color-off);
  }
</style>
