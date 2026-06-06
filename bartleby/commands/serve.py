"""bartleby serve — sync the SvelteKit UI to ~/.bartleby/serve and run it.

The UI source ships inside the package at ``bartleby/web``. On every invocation
we sync that tree into ``~/.bartleby/serve/`` — top-level config files are
copied; ``src/`` is handled one of two ways depending on how Bartleby is
installed:

- **Source checkout / editable install** — ``src/`` is *symlinked* so vite's HMR
  picks up edits in the repo live without a restart.
- **Installed package** (``uv tool install`` / ``pip install``) — the packaged
  ``src/`` is read-only, so it is *copied* instead. There's nothing to live-edit,
  and symlinking a site-packages tree would be pointless.

``node_modules/``, ``.svelte-kit/``, and ``build/`` in the dest are preserved
between runs so we don't re-install or re-bundle every time.

The dev server reads the active project out of ``~/.bartleby/config.yaml``
and opens its SQLite read-only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from bartleby.config import BARTLEBY_DIR
from bartleby.lib import console


SERVE_DIR = BARTLEBY_DIR / "serve"

# The SvelteKit UI is packaged as data inside the ``bartleby`` package, so it
# resolves the same way from a checkout or an installed wheel. ``files()``
# returns a Traversable; the ``str()`` round-trip coerces it to a real ``Path``
# (we need ``.parents`` below and symlink/copy ops, not just Traversable reads).
WEB_SRC = Path(str(files("bartleby"))) / "web"

# Directories vite/npm write into; we leave them untouched between runs.
_PRESERVED = {"node_modules", ".svelte-kit", "build"}
# Things we symlink back into the repo for live HMR (checkout only). Everything
# else (config manifests, the html/css shell) is copied.
_SYMLINKED = {"src"}


def _is_source_checkout() -> bool:
    """True when ``web/`` is part of a live checkout we can symlink into.

    In a source checkout / editable install the package sits at the repo root
    next to ``pyproject.toml``; in an installed wheel it lives under
    ``site-packages`` with no such sibling.
    """
    return (WEB_SRC.parents[1] / "pyproject.toml").exists()


def _require_node() -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        console.error(
            "`bartleby serve` needs node + npm on PATH. "
            "Install Node.js (https://nodejs.org) and try again."
        )
        sys.exit(1)


def _sync_web(src: Path, dst: Path, *, symlink_src: bool = True) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Web source not found at {src}.")
    dst.mkdir(parents=True, exist_ok=True)

    # Clear stale entries (anything not in _PRESERVED). We rebuild them below.
    for entry in dst.iterdir():
        if entry.name in _PRESERVED:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        else:
            shutil.rmtree(entry)

    for entry in src.iterdir():
        if entry.name in _PRESERVED:
            continue
        target = dst / entry.name
        if entry.name in _SYMLINKED and symlink_src:
            target.symlink_to(entry.resolve())
        elif entry.is_file():
            shutil.copy2(entry, target)
        else:
            shutil.copytree(entry, target)


def _needs_install(dst: Path) -> bool:
    # npm writes node_modules/.package-lock.json on every install; comparing
    # against it is more reliable than the directory mtime (which doesn't
    # always change cross-platform when files inside are updated).
    marker = dst / "node_modules" / ".package-lock.json"
    if not marker.exists():
        return True
    return (dst / "package.json").stat().st_mtime > marker.stat().st_mtime


def main() -> None:
    _require_node()
    console.splash()
    checkout = _is_source_checkout()
    console.info(f"Syncing UI from {WEB_SRC} → {SERVE_DIR}…")
    _sync_web(WEB_SRC, SERVE_DIR, symlink_src=checkout)

    if _needs_install(SERVE_DIR):
        console.info("Installing npm dependencies (one-time)…")
        subprocess.run(["npm", "install"], cwd=SERVE_DIR, check=True)

    console.big("Starting Bartleby UI (vite will print the URL below)…")
    os.chdir(SERVE_DIR)
    # Replace this process so Ctrl-C goes straight to npm/vite.
    os.execvp("npm", ["npm", "run", "dev"])
