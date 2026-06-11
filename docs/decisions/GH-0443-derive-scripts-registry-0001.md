# GH-0443 — derive the skill SCRIPTS registry from the package directory

Issue #443 (tier-2 dry-sweep judgment bet) replaced the hand-maintained 23-name
`SCRIPTS` tuple in `bartleby/commands/skill.py` with a derivation over the
`bartleby.skill_scripts` package: the non-underscore modules, sorted.

## What changed

`SCRIPTS` is now

```python
SCRIPTS = tuple(
    sorted(
        m.name
        for m in pkgutil.iter_modules(skill_scripts.__path__)
        if not m.name.startswith("_")
    )
)
```

The package `__init__.py` is empty, so this costs no extra import. The
`UNKNOWN_SKILL` / `MISSING_SKILL` JSON envelopes and `_print_help` are byte-for-byte
unchanged — only the *source* of the name set moved from a literal to the directory.

## Why deriving is correct

The package exists solely to hold skill scripts. The literal tuple was, element for
element, the directory listing minus its two shared-helper modules (`_common.py`,
`_tags.py`) — both of which already follow the underscore convention. Every new
script previously required a parallel edit to this tuple or it silently failed
dispatch with `UNKNOWN_SKILL`; deriving removes that duplicate source of truth.

Dispatch remains an **allowlist**: a name not present in the derived tuple still gets
the `UNKNOWN_SKILL` JSON error, so the derivation does not open the door to arbitrary
module / dotted-path injection — underscore exclusion plus "must be a real module in
this one package" is the whole gate.

## What is no longer handled (the trade)

- A module dropped into `bartleby/skill_scripts/` can no longer be kept out of agent
  dispatch by omitting it from a registry — exclusion now requires the underscore
  prefix. Acceptable: the package holds only skill scripts, and the underscore
  convention for non-script modules already held.
- The stderr help listing's hand-grouped ordering is given up for alphabetical.
  `SKILL.md` is the agent-facing reference for what to call, so that ordering carried
  no weight.

## How the test now guards the derivation

`tests/test_skill_flag_conventions.py` previously kept its *own* hand-copied `SCRIPTS`
roster and `test_guard_roster_matches_dispatcher` asserted the two hand-copies agreed
(the #411 blind-spot guard). That recreated the very duplication #443 removes. The
test now:

- imports `SCRIPTS` from the dispatcher (one source of truth) for its parametrized
  convention guards (#403/#405/#408/#111), and
- `test_dispatcher_roster_derives_from_package` asserts the *derivation property*
  itself — `SCRIPTS` equals the sorted non-underscore modules of
  `bartleby.skill_scripts`, is non-empty, and contains no underscore-prefixed
  names — rather than pinning a hand-copied 23-name list.

So the guard still fails loudly if the dispatcher ever drifts from the package, but it
asserts the rule instead of mirroring a frozen snapshot of its output.

---
*Filed from the 2026-06-11 dry sweep (dead/wet/bloat audit; every item adversarially
verified by an independent defender pass).*
