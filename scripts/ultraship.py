#!/usr/bin/env python
"""The deterministic core of ``/ultraship`` — manifest parsing, structural
validation, and wave (DAG) derivation.

``/ultraship`` (see ``.claude/skills/ultraship/SKILL.md``) assembles a whole
omnibus bundle unattended: a stage-manager fans the sub-issues out to headless
players, integrates them through a serialized merge train, runs a bounded critic
loop, and leaves a finished ``omnibus/vX.Y.Z`` branch plus a morning report. All
of that is *agentic* — it lives in the skill prose the orchestrator executes.

This script is the part that must **not** be left to a model: reading the
omnibus issue body into a structured manifest, proving it is well-formed before
any player is dispatched, and computing the wave schedule. A malformed manifest
discovered at 2 a.m. after three issues landed is the worst outcome, so the
parse/validate path is strict and fail-loud, exactly like ``scripts/release.py``
guards the release path. It has no agents and no git/gh side effects — pure
functions over the issue body string, unit-tested in ``tests/test_ultraship.py``.

The manifest lives in the omnibus body's narrative ``### Sub-issues`` section
(which the #297/#265 tracking regime preserves), one strictly-parsed block per
sub-issue::

    ### Sub-issues
    - #236 — *(bug)* PER_WORKER_GB undercounts parse-worker RSS.
      - **objective:** per-worker estimate tracks measured peak RSS; no swap freeze
      - **touches:** `bartleby/scribe/sizing.py`, `bartleby/scribe/workers.py`
      - **depends-on:** —

plus one required header line — ``**goal:** <a checkable statement>`` — and an
optional anchored director's-notes block::

    <!-- director-notes:start -->
    X is out of scope; prefer approach Y for #232.
    <!-- director-notes:end -->

Three fields do the work: **objective** (one line; the critic/director detect
scope drift against it), **touches** (declared blast radius; overlap forces
serialization), **depends-on** (hard ordering edges). Parallelism is *derived*
from depends-on + touches-overlap — there is deliberately no ``parallel`` field
to contradict the computed answer.

Usage (the skill pipes the omnibus body in on stdin)::

    gh issue view <omnibus> --json body --jq .body | uv run python scripts/ultraship.py validate
    gh issue view <omnibus> --json body --jq .body | uv run python scripts/ultraship.py plan
    gh issue view <omnibus> --json body --jq .body | uv run python scripts/ultraship.py plan --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable

# A "no value" marker an author writes for an empty depends-on, etc.
_EMPTY_TOKENS = {"", "-", "—", "–", "none", "n/a", "na"}

# The three required per-sub-issue fields, in the order they should be written.
REQUIRED_FIELDS = ("objective", "touches", "depends-on")

_DIRECTOR_NOTES_RE = re.compile(
    r"<!--\s*director-notes:start\s*-->(.*?)<!--\s*director-notes:end\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_SUBISSUE_HEADER_RE = re.compile(r"^-\s+#(\d+)\s*(?:[—–-]\s*)?(.*)$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubIssue:
    """One sub-issue as declared in the ``### Sub-issues`` manifest block.

    ``objective``/``touches``/``depends_on`` are ``None`` when the field line is
    *absent* (a structural error), distinct from a present-but-empty value such
    as ``depends-on: —`` (which parses to ``[]`` and is valid).
    """

    number: int
    label: str
    objective: str | None = None
    touches: list[str] | None = None
    depends_on: list[int] | None = None


@dataclass
class Manifest:
    goal: str | None = None
    director_notes: str = ""
    sub_issues: list[SubIssue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------

def _match_field(line: str, name: str) -> str | None:
    """If ``line`` declares the field ``name`` (``**name:** value``, ``name:
    value``, with or without a leading list bullet), return its value; else
    ``None``.

    Markdown bold/bullet decoration around the key and value is tolerated so an
    author can write the field the way it reads best.
    """
    if ":" not in line:
        return None
    left, right = line.split(":", 1)
    key = left.strip().lstrip("-*+ ").strip().strip("*").strip().lower()
    if key != name.lower():
        return None
    return right.strip().strip("*").strip()


def _split_list(value: str) -> list[str]:
    """Split a comma-separated field value into trimmed, de-backticked items.

    A lone empty-marker (``—``, ``none``, …) yields ``[]``.
    """
    if value.strip().lower() in _EMPTY_TOKENS:
        return []
    items = []
    for raw in value.split(","):
        item = raw.strip().strip("`").strip()
        if item and item.lower() not in _EMPTY_TOKENS:
            items.append(item)
    return items


def _parse_deps(value: str) -> list[int]:
    """Parse a ``depends-on`` value (``#236, #240`` or ``—``) into issue numbers,
    de-duplicated and order-preserving."""
    numbers: list[int] = []
    for token in _split_list(value):
        m = re.search(r"#?(\d+)", token)
        if m:
            n = int(m.group(1))
            if n not in numbers:
                numbers.append(n)
    return numbers


def parse_goal(body: str) -> str | None:
    """Return the omnibus ``goal:`` header line value, or ``None`` if absent.

    Scanned outside the ``### Sub-issues`` section so a sub-issue line can never
    be mistaken for the bundle goal.
    """
    for line in _body_outside_sub_issues(body):
        value = _match_field(line, "goal")
        if value is not None and value.lower() not in _EMPTY_TOKENS:
            return value
    return None


def parse_director_notes(body: str) -> str:
    """Return the text inside the anchored director-notes block, or ``""``."""
    m = _DIRECTOR_NOTES_RE.search(body)
    return m.group(1).strip() if m else ""


def _sub_issues_section_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Return ``(start, end)`` line indices of the body of the ``### Sub-issues``
    section (the lines *after* its heading up to the next heading / EOF), or
    ``None`` if there is no such section."""
    start = None
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and m.group(1).strip().lower().rstrip(":") == "sub-issues":
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if _HEADING_RE.match(lines[j]):
            end = j
            break
    return start, end


def _body_outside_sub_issues(body: str) -> Iterable[str]:
    lines = body.splitlines()
    bounds = _sub_issues_section_bounds(lines)
    if bounds is None:
        yield from lines
        return
    start, end = bounds
    for i, line in enumerate(lines):
        if start <= i < end:
            continue
        yield line


def parse_sub_issues(body: str) -> list[SubIssue]:
    """Parse the ``### Sub-issues`` section into ``SubIssue`` records.

    Returns ``[]`` when the section is absent or empty; the validator turns that
    into a readable error rather than a crash.
    """
    lines = body.splitlines()
    bounds = _sub_issues_section_bounds(lines)
    if bounds is None:
        return []
    start, end = bounds

    issues: list[SubIssue] = []
    current: SubIssue | None = None
    for line in lines[start:end]:
        header = _SUBISSUE_HEADER_RE.match(line)
        if header:
            current = SubIssue(number=int(header.group(1)), label=header.group(2).strip())
            issues.append(current)
            continue
        if current is None:
            continue
        if not line[:1].isspace():
            # A non-indented, non-header line ends the current issue's block.
            if line.strip():
                current = None
            continue
        objective = _match_field(line, "objective")
        if objective is not None:
            current.objective = objective
            continue
        touches = _match_field(line, "touches")
        if touches is not None:
            current.touches = _split_list(touches)
            continue
        depends = _match_field(line, "depends-on")
        if depends is not None:
            current.depends_on = _parse_deps(depends)
    return issues


def parse_manifest(body: str) -> Manifest:
    """Parse an omnibus issue body into a :class:`Manifest` (best-effort; call
    :func:`validate_manifest` to check it is well-formed)."""
    return Manifest(
        goal=parse_goal(body),
        director_notes=parse_director_notes(body),
        sub_issues=parse_sub_issues(body),
    )


# ---------------------------------------------------------------------------
# Validation (pure)
# ---------------------------------------------------------------------------

def validate_manifest(manifest: Manifest) -> list[str]:
    """Return a list of human-readable problems; empty means well-formed.

    Strict and fail-loud — ``run`` aborts before dispatching any player if this
    is non-empty. Checks: a ``goal`` is present; at least one sub-issue; no
    duplicate numbers; every required field present and non-empty (an empty
    ``depends-on`` is allowed); every ``depends-on`` resolves to a declared
    sub-issue; and the dependency graph is acyclic.
    """
    problems: list[str] = []

    if not manifest.goal:
        problems.append("missing omnibus `goal:` header line.")

    if not manifest.sub_issues:
        problems.append("no sub-issues found under a `### Sub-issues` section.")
        return problems

    seen: set[int] = set()
    for issue in manifest.sub_issues:
        if issue.number in seen:
            problems.append(f"#{issue.number} is declared more than once.")
        seen.add(issue.number)

    for issue in manifest.sub_issues:
        if not issue.objective:
            problems.append(f"#{issue.number} is missing required field `objective`.")
        if not issue.touches:
            problems.append(f"#{issue.number} is missing required field `touches`.")
        if issue.depends_on is None:
            problems.append(f"#{issue.number} is missing required field `depends-on`.")

    for issue in manifest.sub_issues:
        for dep in issue.depends_on or []:
            if dep not in seen:
                problems.append(
                    f"#{issue.number} depends-on #{dep}, which is not a sub-issue "
                    "of this omnibus."
                )
            if dep == issue.number:
                problems.append(f"#{issue.number} depends-on itself.")

    cycle = _find_cycle(manifest)
    if cycle:
        chain = " → ".join(f"#{n}" for n in cycle)
        problems.append(f"dependency cycle: {chain}.")

    return problems


def _find_cycle(manifest: Manifest) -> list[int] | None:
    """Return one dependency cycle as a list of issue numbers, or ``None``.

    Edges that point at undeclared issues are ignored here (the missing-target
    check reports those separately), so a cycle report is always a real loop
    among declared sub-issues.
    """
    declared = {i.number for i in manifest.sub_issues}
    deps = {
        i.number: [d for d in (i.depends_on or []) if d in declared]
        for i in manifest.sub_issues
    }
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in deps}
    stack: list[int] = []

    def visit(node: int) -> list[int] | None:
        color[node] = GREY
        stack.append(node)
        for nxt in deps[node]:
            if color[nxt] == GREY:
                return stack[stack.index(nxt):] + [nxt]
            if color[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        color[node] = BLACK
        return None

    for node in sorted(deps):
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None


# ---------------------------------------------------------------------------
# Wave (DAG) derivation (pure)
# ---------------------------------------------------------------------------

def derive_waves(manifest: Manifest) -> list[list[int]]:
    """Schedule the sub-issues into waves runnable in order.

    Two constraints, both honored: a ``depends-on`` edge forces its target into a
    strictly earlier wave; and two issues whose ``touches`` sets overlap are
    serialized into different waves (the blast radii collide, so they cannot run
    in parallel). Within a wave every issue is independent and conflict-free.

    Greedy and deterministic: each round takes the dependency-ready issues in
    ascending number order and packs them into the wave, skipping any that would
    overlap a file already claimed by an earlier member of the same wave (those
    fall to the next wave). Call :func:`validate_manifest` first — this assumes an
    acyclic, resolvable graph.
    """
    deps = {i.number: set(i.depends_on or []) for i in manifest.sub_issues}
    touches = {i.number: set(i.touches or []) for i in manifest.sub_issues}
    placed: set[int] = set()
    remaining = set(deps)
    waves: list[list[int]] = []

    while remaining:
        ready = sorted(n for n in remaining if deps[n] <= placed)
        if not ready:
            # Should be unreachable on a validated manifest; guard anyway.
            raise ValueError("unresolvable dependencies (validate the manifest first)")
        wave: list[int] = []
        claimed: set[str] = set()
        for n in ready:
            if touches[n] & claimed:
                continue
            wave.append(n)
            claimed |= touches[n]
        waves.append(wave)
        placed |= set(wave)
        remaining -= set(wave)
    return waves


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def format_plan(manifest: Manifest, waves: list[list[int]]) -> str:
    """Render the validated manifest + wave schedule as readable text for the
    ``plan`` PAUSE."""
    by_number = {i.number: i for i in manifest.sub_issues}
    out: list[str] = []
    out.append(f"goal: {manifest.goal}")
    if manifest.director_notes:
        out.append("")
        out.append("director's notes:")
        for line in manifest.director_notes.splitlines():
            out.append(f"  {line}")
    out.append("")
    out.append(f"{len(manifest.sub_issues)} sub-issues in {len(waves)} wave(s):")
    for w, wave in enumerate(waves, start=1):
        parallel = " (parallel)" if len(wave) > 1 else ""
        out.append(f"  wave {w}{parallel}:")
        for n in wave:
            issue = by_number[n]
            dep_note = (
                f"  [after {', '.join(f'#{d}' for d in issue.depends_on)}]"
                if issue.depends_on
                else ""
            )
            out.append(f"    #{n} — {issue.label}{dep_note}")
    return "\n".join(out)


def manifest_to_dict(manifest: Manifest, waves: list[list[int]]) -> dict:
    return {
        "goal": manifest.goal,
        "director_notes": manifest.director_notes,
        "sub_issues": [
            {
                "number": i.number,
                "label": i.label,
                "objective": i.objective,
                "touches": i.touches,
                "depends_on": i.depends_on,
            }
            for i in manifest.sub_issues
        ],
        "waves": waves,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_body(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse, validate, and schedule an /ultraship omnibus manifest.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "Check the manifest is well-formed; exit non-zero if not."),
        ("plan", "Validate, then print the wave schedule."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "body", nargs="?", default=None,
            help="Path to the omnibus body file (default: read stdin).",
        )
        if name == "plan":
            p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    manifest = parse_manifest(_read_body(args.body))
    problems = validate_manifest(manifest)
    if problems:
        print("error: manifest is not well-formed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if args.command == "validate":
        print(f"ok: {len(manifest.sub_issues)} sub-issues, goal present.")
        return 0

    waves = derive_waves(manifest)
    if args.json:
        print(json.dumps(manifest_to_dict(manifest, waves), indent=2))
    else:
        print(format_plan(manifest, waves))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
