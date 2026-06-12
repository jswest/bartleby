# `ship` / `ultraship` are vendored from a drawer, not hand-authored here.

The two skill dirs `.claude/skills/ship/` and `.claude/skills/ultraship/` are now
**pressed copies** of a single source that lives in a local skill drawer
(`~/Code/workflows/skills/`), copied in verbatim by `signet`. They were
previously hand-maintained in this repo; the same two skills now serve multiple
repos, so one drawer copy is the source of truth and each repo carries a
byte-identical, checked-in copy. Everything bartleby-specific the skills need —
test command, gate agent, worktree scheme, decision paths, the omnibus version
scheme, the live-corpus redline, the playwright settings — moved out of the prose
and into per-repo `.claude/ship.toml` / `.claude/ultraship.toml`, which the press
never touches (they sit one level up from the skill dir). The pressed SKILL.md is
repo-agnostic and reads its specifics from config.

Three load-bearing repo-side bits make the arrangement honest:

- **The configs are committed.** `.gitignore` re-includes `!.claude/ship.toml` and
  `!.claude/ultraship.toml` against the `.claude/*` blanket ignore — without that
  whitelist they'd be untracked and every fresh clone would route into the config
  wizard with empty values (and an unattended `ultraship` player would inherit an
  *empty* `player_guardrails`, crossing the exact redline the config exists to
  prevent).
- **The `.stamp` is gitignored.** Each press leaves `.claude/skills/<skill>/.stamp`
  naming an absolute drawer path + timestamp — machine-local noise to any other
  clone — so `.claude/skills/*/.stamp` is ignored. A clone without the drawer has
  no stamp and correctly self-skips.
- **A drift gate rides the test suite.** `tests/test_skill_drift.py` runs
  `signet press <skill> --into . --check` and fails if a vendored copy has drifted
  from the drawer (a hand-edit here, or a drawer that advanced without a re-press).
  It **skips** when `signet` or the drawer is absent — the clone / CI / teammate
  case — because the only machine that can be out of sync with the drawer is one
  that has it. This puts the gate in the existing `uv run pytest` flow rather than
  adding a separate commit hook.

The vendored `ultraship` ships its deterministic manifest core as
`.claude/skills/ultraship/ultraship.py` (pressed alongside the SKILL.md), which
**supersedes the old `scripts/ultraship.py`** (GH-0244): the two were
function-for-function identical — only the module docstring differed (generalized,
pure-`python3`) — so the old copy was deleted rather than left as a dormant twin,
`tests/test_ultraship.py` repointed at the pressed path (same functions, still
green), and CONTRIBUTING's references moved. The GH-0244 entry is left as-is per
the never-edit-old-decisions rule; this note records the relocation.

`release` was deliberately left repo-local — it is intrinsically bartleby-specific
and has no second consumer to justify generalizing. Generalizing the SKILL.md
content itself is drawer-side work; this repo's job is to consume the pressed
skills cleanly and capture its own specifics in the two `.toml` files (issue #530).
