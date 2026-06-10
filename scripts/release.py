#!/usr/bin/env python
"""Cut a Bartleby release.

Versioning scheme — ``v0.<SCHEMA_VERSION>.<patch>``:

* **MINOR == SCHEMA_VERSION** (read straight from ``bartleby/db/schema.py``). It
  increments monotonically, so the version number self-documents the schema:
  ``v0.8.3`` means "schema 8, third patch". A minor bump has two dispositions.
  *Additive* — every crossed version has a ``_UPGRADES`` chain entry
  (``bartleby/db/upgrades.py``): existing corpora run
  ``bartleby project upgrade <name>`` and the notes say so. *Breaking* — some
  step has no chain entry: the notes order a full re-ingest.
* **PATCH** increments for every release at the same schema; resets to ``0`` on a
  schema bump.
* **MAJOR** is a human call (a real 1.0, a ground-up rewrite) — never automated
  here. It is carried forward from the last tag unchanged.

The constant *is* the minor, so computing the next version needs no diffing —
just the schema integer and the last tag.

Usage (from the repo root)::

    uv run python scripts/release.py            # dry run: print what would happen
    uv run python scripts/release.py --tag       # create the tag locally
    uv run python scripts/release.py --tag --push  # tag, push, publish GH release

The release is cut from whatever ``HEAD`` points at; run it from ``main`` after
the change has merged. A drift guard refuses to tag when the schema DDL changed
but ``SCHEMA_VERSION`` did not.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
SCHEMA_PATH = "bartleby/db/schema.py"

from bartleby.db.upgrades import _UPGRADES  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def parse_schema_module(source: str) -> tuple[int, str]:
    """Return ``(SCHEMA_VERSION, DDL)`` parsed from a ``schema.py`` source string.

    Uses the AST rather than regex so a reformatted DDL or a moved constant
    doesn't silently break the release tooling.
    """
    tree = ast.parse(source)
    version: int | None = None
    ddl: str | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "SCHEMA_VERSION":
                version = ast.literal_eval(node.value)
            elif target.id == "DDL":
                ddl = ast.literal_eval(node.value)
    if version is None:
        raise ValueError(f"SCHEMA_VERSION not found in {SCHEMA_PATH}")
    if ddl is None:
        raise ValueError(f"DDL not found in {SCHEMA_PATH}")
    return version, ddl


def parse_version_tag(tag: str) -> tuple[int, int, int]:
    """``"v0.7.2"`` -> ``(0, 7, 2)``. Accepts an optional leading ``v``."""
    core = tag.lstrip("v").split("+", 1)[0]
    parts = core.split(".")
    if len(parts) != 3:
        raise ValueError(f"not a vMAJOR.MINOR.PATCH tag: {tag!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def compute_next_version(schema_version: int, last_tag: str | None) -> str:
    """Derive the next ``MAJOR.MINOR.PATCH`` string from the schema + last tag.

    No prior tag -> baseline ``0.<schema>.0``. Schema moved since the last tag
    -> reset patch to 0 at the new minor. Otherwise bump patch.
    """
    if last_tag is None:
        return f"0.{schema_version}.0"
    major, minor, patch = parse_version_tag(last_tag)
    if schema_version != minor:
        return f"{major}.{schema_version}.0"
    return f"{major}.{minor}.{patch + 1}"


def schema_moved(schema_version: int, last_tag: str | None) -> bool:
    """True when this release crosses a schema boundary (re-ingest required)."""
    if last_tag is None:
        return False
    return parse_version_tag(last_tag)[1] != schema_version


def upgrade_covers(schema_from: int, schema_to: int) -> bool:
    """True when the ``_UPGRADES`` chain spans every crossed version.

    An *additive* bump has a chain entry for each step ``v -> v+1`` between the
    old and new schema, so existing corpora run ``bartleby project upgrade``
    rather than re-ingesting. A single missing step makes the whole bump
    breaking (re-ingest). A non-forward move (``schema_from >= schema_to``) is
    never an additive upgrade.
    """
    if schema_from >= schema_to:
        return False
    return all(v in _UPGRADES for v in range(schema_from, schema_to))


def check_drift(old_source: str, new_source: str) -> str | None:
    """Compare two ``schema.py`` sources; return an error string if they drift.

    Hard error (returned non-None): the DDL changed but ``SCHEMA_VERSION`` did
    not — a schema change that forgot to bump the constant, which would ship a
    breaking change mislabeled as a safe patch. The caller refuses to tag.

    A ``SCHEMA_VERSION`` bump with no DDL change is allowed (e.g. a constraint
    expressed outside the DDL string) and returns None.
    """
    old_version, old_ddl = parse_schema_module(old_source)
    new_version, new_ddl = parse_schema_module(new_source)
    if old_ddl != new_ddl and old_version == new_version:
        return (
            f"schema DDL changed but SCHEMA_VERSION is still {new_version}. "
            "Bump SCHEMA_VERSION in "
            f"{SCHEMA_PATH} before releasing (a schema change forces a re-ingest)."
        )
    return None


def build_release_notes(
    log_lines: list[str],
    *,
    schema_from: int | None,
    schema_to: int | None,
) -> str:
    """Assemble GitHub Release notes from the git log, with a schema banner
    prepended on a schema bump.

    An *additive* bump (``_UPGRADES`` covers every crossed version) tells users
    to run ``bartleby project upgrade <name>``; a *breaking* bump orders a full
    re-ingest.
    """
    parts: list[str] = []
    if schema_from is not None and schema_to is not None and schema_from != schema_to:
        remedy = (
            "Existing corpora upgrade in place: run `bartleby project upgrade <name>`."
            if upgrade_covers(schema_from, schema_to)
            else "Existing corpora must be re-ingested."
        )
        parts.append(
            f"⚠️ **This release changes the database schema "
            f"({schema_from} → {schema_to}). {remedy}**\n"
        )
    parts.append("## Changes\n")
    if log_lines:
        parts.extend(f"- {line}" for line in log_lines)
    else:
        parts.append("- Baseline release.")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Git / gh side-effects
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def last_release_tag() -> str | None:
    """The most recent ``v0.*`` tag reachable from HEAD, or None if untagged."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def schema_source_at(ref: str) -> str:
    return _git("show", f"{ref}:{SCHEMA_PATH}")


def commits_since(ref: str | None) -> list[str]:
    spec = f"{ref}..HEAD" if ref else "HEAD"
    out = _git("log", spec, "--no-merges", "--pretty=format:%s")
    return [line for line in out.splitlines() if line.strip()]


def working_tree_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut a Bartleby release.")
    parser.add_argument(
        "--tag", action="store_true",
        help="Create the git tag locally (default is a dry run).",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Push the tag and publish a GitHub Release (implies --tag).",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Permit a tag on a dirty working tree (refused by default).",
    )
    args = parser.parse_args(argv)
    do_tag = args.tag or args.push

    schema_source = (REPO_ROOT / SCHEMA_PATH).read_text()
    schema_version, _ = parse_schema_module(schema_source)
    last_tag = last_release_tag()

    # Drift guard — refuse a mislabeled schema change.
    if last_tag is not None:
        problem = check_drift(schema_source_at(last_tag), schema_source)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

    next_version = compute_next_version(schema_version, last_tag)
    new_tag = f"v{next_version}"
    moved = schema_moved(schema_version, last_tag)
    schema_from = parse_version_tag(last_tag)[1] if moved else None

    # The baseline release gets a clean "Baseline release." note rather than the
    # project's entire history; subsequent releases list commits since the tag.
    commits = commits_since(last_tag) if last_tag is not None else []
    if last_tag is not None and not commits:
        print(f"error: no commits since {last_tag}; nothing to release.", file=sys.stderr)
        return 1

    notes = build_release_notes(
        commits, schema_from=schema_from, schema_to=schema_version if moved else None,
    )

    if moved:
        disposition = (
            "  (bumped → upgrade in place)"
            if upgrade_covers(schema_from, schema_version)
            else "  (bumped → re-ingest)"
        )
    else:
        disposition = ""
    print(f"Last tag:     {last_tag or '(none)'}")
    print(f"Schema:       {schema_version}{disposition}")
    print(f"Next release: {new_tag}")
    print()
    print(notes)

    if not do_tag:
        print("Dry run — pass --tag to create the tag, --push to publish.", file=sys.stderr)
        return 0

    if working_tree_dirty() and not args.allow_dirty:
        print("error: working tree is dirty; commit or stash first (or --allow-dirty).",
              file=sys.stderr)
        return 1
    branch = current_branch()
    if branch != "main":
        print(f"warning: releasing from '{branch}', not 'main'.", file=sys.stderr)

    _git("tag", "-a", new_tag, "-m", f"Release {new_tag}")
    print(f"Created tag {new_tag}.", file=sys.stderr)

    if args.push:
        _git("push", "origin", new_tag)
        subprocess.run(
            ["gh", "release", "create", new_tag, "--title", new_tag, "--notes", notes],
            cwd=REPO_ROOT, check=True,
        )
        print(f"Pushed {new_tag} and published the GitHub Release.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
