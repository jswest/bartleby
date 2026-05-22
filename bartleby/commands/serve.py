"""bartleby serve — sync the SvelteKit UI to ~/.bartleby/serve and run it.

The UI source lives in ``web/`` in the repo. On every invocation we sync that
tree into ``~/.bartleby/serve/`` — top-level config files are copied; ``src/``
is symlinked so vite's HMR picks up edits in the repo live without a restart.
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
from pathlib import Path

from bartleby.config import BARTLEBY_DIR
from bartleby.lib import console


SERVE_DIR = BARTLEBY_DIR / "serve"

# Walk up from this file to the repo root, where ``web/`` lives.
WEB_SRC = Path(__file__).resolve().parents[2] / "web"

# Directories vite/npm write into; we leave them untouched between runs.
_PRESERVED = {"node_modules", ".svelte-kit", "build"}
# Things we symlink back into the repo for live HMR. Everything else (config
# manifests, the html/css shell) is copied.
_SYMLINKED = {"src"}


def _require_node() -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        console.error(
            "`bartleby serve` needs node + npm on PATH. "
            "Install Node.js (https://nodejs.org) and try again."
        )
        sys.exit(1)


def _sync_web(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(
            f"Web source not found at {src}. "
            "Are you running from a source checkout?"
        )
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
        target = dst / entry.name
        if entry.name in _SYMLINKED:
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
    console.info(f"Syncing UI from {WEB_SRC} → {SERVE_DIR}…")
    _sync_web(WEB_SRC, SERVE_DIR)

    if _needs_install(SERVE_DIR):
        console.info("Installing npm dependencies (one-time)…")
        subprocess.run(["npm", "install"], cwd=SERVE_DIR, check=True)

    console.big("Starting Bartleby UI (vite will print the URL below)…")
    os.chdir(SERVE_DIR)
    # Replace this process so Ctrl-C goes straight to npm/vite.
    os.execvp("npm", ["npm", "run", "dev"])
