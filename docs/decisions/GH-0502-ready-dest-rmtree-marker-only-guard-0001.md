# `ready --dest` rmtrees only dirs carrying our `.bartleby-skill` marker, not any with a SKILL.md (issue #502)

> Source: [#502](https://github.com/jswest/bartleby/issues/502)

`bartleby ready` wipes its destination (`shutil.rmtree`) before copying the packaged skill in, guarded by `_looks_like_skill_dir`, which accepted *either* a `SKILL.md` **or** the `.bartleby-skill` marker. But `SKILL.md` is the universal filename for *any* Claude skill: pointing `--dest` at a foreign skill (another tool's `~/.claude/skills/<name>`) sailed past the guard and silently deleted it. The marker is the only thing that actually proves an install is ours — `_install` stamps it via `_write_marker` on every copy — so the bare-`SKILL.md` arm of the guard was never a real ownership signal, just a footgun.

So `_looks_like_skill_dir` becomes `_is_ours`, keyed solely on the `.bartleby-skill` marker. The refusal is hard (no `--force` override), matching how the guard already ignored `force` before — a non-empty `--dest` without our marker refuses outright, and the message now says it's "not ours" and names the marker, explicitly noting that a stray `SKILL.md` alone doesn't make it ours. Empty dirs and our own marked dirs still install/refresh as before (`force` still governs the up-to-date no-op, not this safety check). No schema change.
