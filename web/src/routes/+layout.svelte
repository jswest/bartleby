<script>
  import "../app.css";
  export let data;
</script>

<header>
  <h2 class="display"><a href="/">Bartleby</a></h2>
  <h2 class="display project">{data.project}</h2>
</header>

<main>
  <slot />
</main>

<style>
  header {
    padding: 1rem;
    border-bottom: 1px solid #ccc;
  }
  header a {
    color: inherit;
    text-decoration: none;
    font-weight: bold;
  }
  .project {
    color: var(--color-off);
    font-size: 1rem;
  }
  main {
    padding: 1rem;
  }
</style>
