"""`bartleby skill <name> [args]` — run one of the skill scripts.

This is the agent-facing surface. The scripts live in
``bartleby.skill_scripts.*`` and emit JSON to stdout. The dispatcher exists
so harnesses don't have to know which Python interpreter has ``bartleby``
installed — they just need ``bartleby`` on ``PATH``.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import sys

from bartleby import skill_scripts


# The dispatchable scripts are exactly the non-underscore modules of the
# ``bartleby.skill_scripts`` package (``_common``/``_tags`` are shared helpers,
# not skills). Derive the allowlist from the package directory so a new script
# is dispatchable on drop-in, with no parallel edit here. Sorted for a stable
# help/error ordering.
SCRIPTS = tuple(
    sorted(
        m.name
        for m in pkgutil.iter_modules(skill_scripts.__path__)
        if not m.name.startswith("_")
    )
)


def _print_help(stream=sys.stderr) -> None:
    print("usage: bartleby skill <name> [args...]", file=stream)
    print("", file=stream)
    print("Available scripts:", file=stream)
    for s in SCRIPTS:
        print(f"  {s}", file=stream)
    print("", file=stream)
    print("Pass --help to a specific script for its arguments, e.g.:", file=stream)
    print("  bartleby skill search --help", file=stream)


def dispatch(argv: list[str]) -> None:
    if argv and argv[0] in ("-h", "--help"):
        _print_help(sys.stdout)
        return

    if not argv:
        # No script named: emit the JSON error envelope (like every other
        # skill error path) and send the usage help to stderr only.
        _print_help(sys.stderr)
        payload = {
            "error": f"No skill script given. Available: {', '.join(SCRIPTS)}.",
            "code": "MISSING_SKILL",
        }
        json.dump(payload, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.exit(1)

    name, *rest = argv
    if name not in SCRIPTS:
        # Emit JSON error envelope so the agent can parse it.
        payload = {
            "error": f"Unknown skill script {name!r}. "
                     f"Available: {', '.join(SCRIPTS)}.",
            "code": "UNKNOWN_SKILL",
        }
        json.dump(payload, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.exit(1)

    try:
        module = importlib.import_module(f"bartleby.skill_scripts.{name}")
    except ModuleNotFoundError as e:
        # A stale editable/tool install can be missing a dependency the
        # script's import chain pulls at module load (apsw, sqlite_vec, ...),
        # which fails here — before skill_runner.run()'s STALE_INSTALL arm
        # exists to catch it. Emit the same envelope inline; importing
        # skill_runner for a shared helper would walk the same broken chain.
        payload = {
            "error": (
                f"Missing dependency: {e.name or e}. The running environment "
                "doesn't have a module the installed code now imports — most "
                "likely a stale `uv tool install --editable` that hasn't "
                "picked up a dependency added since you installed. Run "
                "`uv tool install --reinstall <path-or-package>` (or "
                "otherwise re-sync your environment) and retry. "
                f"Original error: {type(e).__name__}: {e}"
            ),
            "code": "STALE_INSTALL",
        }
        json.dump(payload, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.exit(1)
    module.main(rest)
