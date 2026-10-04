# `finding read` / `export` take `finding:<N>`, not bare ints (issue #742)

> Source: [#742](https://github.com/jswest/bartleby/issues/742)

the `finding` subcommand group was inconsistent about id shapes: the annotation
subcommands (`annotate`, `annotations`, `delete-annotation`) already took
type-tagged ids via `prefixed_int` (the #573/#624 convention that made bare ids
structurally unambiguous), but `read` and `export` still accepted a bare int
positional — so `bartleby finding read 5` worked while
`bartleby finding annotate 5 …` failed. That split was a latent trap: an agent
reading one command's help would assume the bare form and then hit a hard
rejection on a sibling subcommand of the same group. Fix: give `read` and
`export` the same `type=prefixed_int("finding")` / `metavar="finding:<N>"`
positional the annotation subcommands use, so the whole group speaks one id
shape. The `prefixed_int` import had to move above the `read`/`export` parsers (it
sat below them, in scope only for the annotation subcommand that first used it).
The command functions in `commands/finding.py` are untouched — they already take a
bare `int`; the argparse layer yields the bare int, so nothing downstream changes.
This is a cutover of the human/agent CLI surface to the convention the rest of the
group already enforced, not a new one; there's nothing to keep both.
