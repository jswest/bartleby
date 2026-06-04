<script>
  import "../app.css";
  import { page } from "$app/stores";
  import { marked } from "marked";
  export let data;

  marked.use({
    hooks: {
      postprocess: (html) =>
        html.replace(/<a /g, '<a target="_blank" rel="noopener noreferrer" '),
    },
  });

  // A nav link is "active" if the current path matches exactly OR is a child
  // of the link's target (so /findings/15 lights up the Findings nav too).
  // Home is exact-match only — otherwise it'd be active everywhere.
  $: path = $page.url.pathname;
  function isActive(href) {
    if (href === "/") return path === "/";
    return path === href || path.startsWith(href + "/");
  }
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
    gap: 1.5rem;
    padding: 0.75rem 1.25rem;
  }
  .brand {
    font-size: 1.25rem;
    font-weight: 900;
    color: var(--color-token);
    text-decoration: none;
  }
  .links {
    display: flex;
    gap: 1.25rem;
  }
  .links a {
    color: var(--color-off-light);
    text-decoration: none;
    font-family: var(--font-sans);
    font-size: 1rem;
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
    font-size: 1rem;
  }
  main {
    /* Offset the fixed header so content doesn't slide under it. */
    padding: 4rem 1.25rem 2rem;
  }
</style>
