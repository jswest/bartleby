"""`bartleby ready` — install or refresh the skill into an agent harness.

The harness-facing skill (SKILL.md + README.md) ships as package data under
``bartleby/skill``, so this command works from an installed tool, not just a
checkout. It stamps that skill into ``~/.claude/skills/bartleby`` (or
``--dest``), wiping any prior copy first so the folder can't nest one level
too deep — the classic ``cp -r`` footgun the README used to warn about.

"Latest" is decided by a content hash over the skill's files, not the version
number: ``SKILL.md`` is edited constantly off-tag, so two builds at the same
version can carry different skills. A small ``.bartleby-skill`` marker written
into the destination records that hash plus the bartleby version that produced
it, so we can report ``up to date`` vs ``stale`` and show a readable version
line.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import bartleby
from bartleby import __version__
from bartleby.lib import console

DEFAULT_DEST = Path.home() / ".claude" / "skills" / "bartleby"

# Hidden marker dropped alongside SKILL.md so we can recognise our own installs
# and report the version that produced them. Harnesses ignore unknown files.
MARKER_NAME = ".bartleby-skill"


def _source_dir() -> Path:
    """The packaged skill directory — resolves for a checkout and a wheel alike."""
    return Path(bartleby.__file__).resolve().parent / "skill"


def _hash_dir(root: Path) -> str | None:
    """SHA-256 over the directory's files (name + bytes), sorted for stability.

    The marker is excluded so an install's own version stamp doesn't perturb
    the content identity it records. Returns ``None`` when ``root`` is absent.
    """
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.name == MARKER_NAME or not path.is_file():
            continue
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _read_marker(dest: Path) -> dict:
    marker = dest / MARKER_NAME
    if not marker.is_file():
        return {}
    try:
        return json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _looks_like_skill_dir(dest: Path) -> bool:
    """A guard against ``rmtree``-ing an unrelated directory passed as --dest."""
    return (dest / "SKILL.md").is_file() or (dest / MARKER_NAME).is_file()


def _write_marker(dest: Path, src_hash: str) -> None:
    """Stamp the destination with this tool's version and the content hash.

    Cheap enough to call on its own when only the version stamp is stale —
    the skill content is already byte-identical, so a full reinstall would be
    wasted work.
    """
    (dest / MARKER_NAME).write_text(
        json.dumps({"version": __version__, "hash": src_hash}) + "\n"
    )


def _install(src: Path, dest: Path, src_hash: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    _write_marker(dest, src_hash)


def main(*, dest: Path | None = None, check: bool = False, force: bool = False) -> None:
    src = _source_dir()
    src_hash = _hash_dir(src)
    if src_hash is None:
        console.error(f"Skill source not found at {src}.")
        sys.exit(1)

    dest = dest or DEFAULT_DEST
    dest_hash = _hash_dir(dest)
    installed_version = _read_marker(dest).get("version")
    up_to_date = dest_hash is not None and dest_hash == src_hash

    if check:
        if up_to_date:
            console.complete(f"Skill is up to date (v{installed_version or '?'}) at {dest}.")
            return
        if dest_hash is None:
            console.warn(f"Skill is not installed at {dest}. Run `bartleby ready`.")
        elif installed_version == __version__:
            console.warn(
                f"Skill content has drifted from the packaged copy (v{__version__}). "
                "Run `bartleby ready`."
            )
        else:
            console.warn(
                f"Skill is stale (installed v{installed_version or '?'} → "
                f"packaged v{__version__}). Run `bartleby ready`."
            )
        sys.exit(1)

    if up_to_date and not force:
        if installed_version != __version__:
            # Content is byte-identical but the marker predates this tool
            # version (the common case — SKILL.md rarely changes across a bump).
            # Refresh just the stamp so the version line stays truthful and
            # `--check`, which reads it back, agrees. No full reinstall needed.
            _write_marker(dest, src_hash)
            console.big(
                f"Updated skill marker v{installed_version or '?'} → v{__version__} at {dest}"
            )
        else:
            console.complete(f"Skill already up to date (v{__version__}) at {dest}.")
        return

    if dest.exists() and any(dest.iterdir()) and not _looks_like_skill_dir(dest):
        console.error(
            f"{dest} exists and doesn't look like a skill directory (no SKILL.md). "
            "Refusing to overwrite it — pass a different --dest."
        )
        sys.exit(1)

    _install(src, dest, src_hash)

    if dest_hash is None:
        console.big(f"Installed skill v{__version__} → {dest}")
    elif installed_version and installed_version != __version__:
        console.big(f"Updated skill v{installed_version} → v{__version__} at {dest}")
    else:
        console.big(f"Refreshed skill v{__version__} at {dest}")
    console.warn("Restart your harness so it reloads the skill.")
