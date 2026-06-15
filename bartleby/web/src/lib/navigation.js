// True when an in-flight navigation is a real search submission — going TO
// /search with a non-empty ?q= (a query submit, a scan step, or pagination,
// all of which carry q). The search page shows "Searching…" when this is true;
// the global layout indicator shows "Loading…" when it is false. They are
// complements, so the rule lives in one place: a one-sided edit would otherwise
// produce a double indicator or a silent gap.
export function isSearchNavigation(navigating) {
  return (
    !!navigating &&
    navigating.to?.url?.pathname?.startsWith("/search") &&
    !!navigating.to?.url?.searchParams?.get("q")
  );
}
