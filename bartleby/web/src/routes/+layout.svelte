<script context="module">
  import { marked } from "marked";
  import DOMPurify from "isomorphic-dompurify";

  // Every `{@html marked.parse(...)}` site (summaries, finding bodies, markdown
  // sources) flows through this one global parser. Summaries are model-generated
  // and finding bodies agent-authored — both untrusted — and `marked` passes raw
  // HTML through, so an embedded <script> would run. Sanitize first (DOMPurify
  // strips scripts/handlers/js: URLs while keeping the cite-chip <button>), then
  // add link attrs to the clean output so target/rel don't depend on the sanitizer.
  //
  // Module scope, not instance: `marked.use` registers globally and *chains*
  // passthrough hooks rather than replacing them, so per-SSR-request registration
  // re-ran the hook once per prior request in a long-lived `serve` — duplicating
  // target/rel and burning CPU. The module body runs once per process.
  marked.use({
    hooks: {
      postprocess: (html) =>
        DOMPurify.sanitize(html).replace(
          /<a /g,
          '<a target="_blank" rel="noopener noreferrer" ',
        ),
    },
  });
</script>

<script>
  import "../app.css";
  import { page } from "$app/stores";
  export let data;

  // A nav link is "active" if the current path matches exactly OR is a child
  // of the link's target (so /findings/15 lights up the Findings nav too).
  // Home is exact-match only — otherwise it'd be active everywhere.
  //
  // isActive must be reactive ($:), not a plain function: Svelte only
  // re-evaluates `class:active={isActive(...)}` when a variable named in the
  // expression changes. A plain function references only itself (constant), so
  // the binding would be computed once and freeze — leaving whichever link was
  // active on first load stuck underlined across client-side navigations.
  $: path = $page.url.pathname;
  $: isActive = (href) =>
    href === "/" ? path === "/" : path === href || path.startsWith(href + "/");
</script>

<header>
  <nav class="bar">
    <a class="brand display" href="/" class:active={isActive("/")}>Bartleby</a>
    <div class="links">
      <a href="/search" class:active={isActive("/search")}>Search</a>
      <a href="/findings" class:active={isActive("/findings")}>Findings</a>
      <a href="/documents" class:active={isActive("/documents")}>Documents</a>
      <a href="/tags" class:active={isActive("/tags")}>Tags</a>
    </div>
    <span class="project display">{data.project}</span>
  </nav>
</header>

<main>
  <slot />
</main>

<style>
  header {
    background-color: var(--color-off);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 10;
    border-bottom: 1px solid var(--color-off-light);
  }
  .bar {
    display: flex;
    align-items: baseline;
    gap: var(--space-xl);
    padding: var(--space-md) var(--space-lg);
  }
  .brand {
    font-size: var(--text-xl);
    font-weight: 900;
    color: var(--color-token);
    text-decoration: none;
  }
  .links {
    display: flex;
    gap: var(--space-lg);
  }
  .links a {
    color: var(--color-off-light);
    text-decoration: none;
    font-family: var(--font-sans);
    font-size: var(--text-base);
    padding-bottom: 2px;
    border-bottom: 2px solid transparent;
  }
  .links a:hover {
    color: var(--color-off-light);
  }
  .links a.active {
    color: var(--color-off-light);
    border-bottom-color: var(--color-off-light);
  }
  .project {
    margin-left: auto;
    color: var(--color-off-light);
    font-size: var(--text-base);
  }
  main {
    /* Offset the fixed header (a fixed structural height) so content doesn't
       slide under it; the horizontal value is the shared page gutter. */
    padding: 4rem var(--space-lg) var(--space-2xl);
  }
</style>
