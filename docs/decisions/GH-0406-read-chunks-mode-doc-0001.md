# `read_chunks` documents that out-of-mode flags are ignored (issue #406)

> Source: [#406](https://github.com/jswest/bartleby/issues/406)

`read_chunks` has three mutually-exclusive modes (`--document`, `--chunks`,
`--around-chunk`). Flags belonging to one mode (e.g. `--window`, `--offset`,
`--limit`) are simply unread when another mode is selected — they are silently
ignored rather than rejected.

The fix is a **doc line, not enforcement code**. A one-sentence caveat was added
to the existing mode-grouped flag documentation near the top of
`bartleby/skill_scripts/read_chunks.py` (the module docstring, which also feeds
the `--help` text via `build_arg_parser("read_chunks", __doc__)`), stating
plainly that flags belonging to the other modes are silently ignored.

This was the decision reached in the plan interview for #406: the out-of-mode
flags are *inert, not unsafe*, and the issue's own acceptance criteria accept the
docstring alternative over rejection logic. Adding a `p.error()` leg would require
explicit-vs-default flag detection (argparse can't otherwise tell a passed default
from an omitted flag) — that's over-building for an inert condition. Behavior is
therefore unchanged; only the docstring/`--help` text gains the caveat. No schema
change. `tests/test_skill_read_chunks.py` (29 tests) still passes unmodified.
