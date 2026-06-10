"""`bartleby skill <name> [args]` — run one of the skill scripts.

This is the agent-facing surface. The scripts live in
``bartleby.skill_scripts.*`` and emit JSON to stdout. The dispatcher exists
so harnesses don't have to know which Python interpreter has ``bartleby``
installed — they just need ``bartleby`` on ``PATH``.
"""

from __future__ import annotations

import importlib
import json
import sys


SCRIPTS = (
    "describe_corpus",
    "list_documents",
    "search",
    "scan",
    "read_chunks",
    "read_document",
    "save_summary",
    "save_finding",
    "edit_finding",
    "delete_finding",
    "merge_findings",
    "list_findings",
    "read_finding",
    "read_tags",
    "add_tag",
    "delete_tag",
    "rename_tag",
    "merge_tags",
    "tag",
    "assign_tag",
    "unassign_tag",
    "extract",
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
    if not argv or argv[0] in ("-h", "--help"):
        _print_help(sys.stdout)
        return

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

    module = importlib.import_module(f"bartleby.skill_scripts.{name}")
    module.main(rest)
