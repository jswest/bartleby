<script>
  import "../app.css";
  import { page } from "$app/stores";
  export let data;

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
      <a href="/findings" class:active={isActive("/findings")}>Findings</a>
      <a href="/documents" class:active={isActive("/documents")}>Documents</a>
    </div>
    <span class="project display">{data.project}</span>
  </nav>
</header>

<main>
  <slot />
</main>

<style>
  header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 10;
    background: #fff;
    border-bottom: 1px solid #d6d4c6;
  }
  .bar {
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    padding: 0.75rem 1.25rem;
  }
  .brand {
    font-size: 1.25rem;
    font-weight: bold;
    color: inherit;
    text-decoration: none;
  }
  .links {
    display: flex;
    gap: 1.25rem;
  }
  .links a {
    color: var(--color-off);
    text-decoration: none;
    font-family: var(--font-sans);
    font-size: 0.95rem;
    padding-bottom: 2px;
    border-bottom: 2px solid transparent;
  }
  .links a:hover {
    color: #222;
  }
  .links a.active {
    color: #222;
    border-bottom-color: var(--color-off);
  }
  .project {
    margin-left: auto;
    color: var(--color-off);
    font-size: 0.95rem;
  }
  main {
    /* Offset the fixed header so content doesn't slide under it. */
    padding: 4rem 1.25rem 2rem;
  }
</style>
