"""Structural guard for the #111 skill-script flag conventions (issue #411).

Nothing previously *enforced* the flag conventions the #111 omnibus settled —
they held only by reviewer vigilance. This test introspects each skill script's
argparse parser and fails the moment a convention drifts:

  - #403 — pagination flags use the shared validators: every ``--limit`` action
    is typed ``positive_int`` and every ``--offset`` action is typed
    ``nonneg_int`` (not bare ``int``).
  - #405 — a script that identifies a *single existing tag* names that selector
    ``--tag`` (never ``--name`` / ``--tag-name``); ``--name`` is reserved for the
    tag being *created* (``add_tag``).
  - #408 — every scope-supporting script exposes ``--in-documents``.
  - #573 (supersedes #111) — *name the value you accept*: an id-valued flag
    ends in ``-id`` (``--document-id`` / ``--finding-id`` / ``--chunk-id``), a
    name/key/path/predicate flag stays bare. The enforceable invariant is a
    biconditional: a flag's option string ends in ``-id`` **iff** its argparse
    ``dest`` ends in ``_id``. Plural/relational id flags whose ``dest`` ends in
    ``_ids`` or another stem (``--documents`` → ``document_ids``, ``--chunks`` →
    ``chunk_ids``, ``--from`` → ``from_ids``, ``--into`` → ``into``,
    ``--around-chunk`` → ``around_chunk``) stay bare by the same rule.

By design these assertions are LITERAL: each names the exact scripts and exact
flags the conventions cover. They are a tripwire, not a re-derivation of the
spec — no plural/arity/comma-list generalization that would turn the guard into
a brittle mirror of every script's full signature.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import re
import subprocess
from pathlib import Path

import pytest

from bartleby import skill_scripts
from bartleby.commands.skill import SCRIPTS
from bartleby.skill_scripts._common import nonneg_int, positive_int

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- parser introspection -------------------------------------------------
#
# Every skill script exposes ``parse_args(argv)`` which builds an
# ``ArgumentParser`` and *immediately* calls ``parser.parse_args(argv)``. To get
# at the parser without supplying (per-script) valid argv, we intercept that
# final call: a patched ``parse_args`` raises with the parser as payload, so the
# build runs but the consume doesn't. Scripts that don't follow this shape (none
# do today) surface as skips rather than silent passes.


class _ParserCaptured(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self.parser = parser


def _parser_for(module_name: str) -> argparse.ArgumentParser:
    mod = importlib.import_module(f"bartleby.skill_scripts.{module_name}")
    if not hasattr(mod, "parse_args"):
        pytest.skip(f"{module_name} has no parse_args to introspect")

    original = argparse.ArgumentParser.parse_args

    def _capture(self, args=None, namespace=None):  # noqa: ANN001
        raise _ParserCaptured(self)

    argparse.ArgumentParser.parse_args = _capture
    try:
        mod.parse_args(None)
    except _ParserCaptured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    pytest.skip(f"{module_name}.parse_args did not call parser.parse_args")


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {s for action in parser._actions for s in action.option_strings}


def _action_for(parser: argparse.ArgumentParser, option: str):
    return next((a for a in parser._actions if option in a.option_strings), None)


# The skill scripts whose argparse parser we introspect are exactly the
# dispatcher's roster (everything but the ``_``-prefixed shared modules); we
# reuse ``SCRIPTS`` imported from the dispatcher rather than re-listing them.

# Scripts that identify a *single existing* tag — the selector must be ``--tag``
# (#405). ``add_tag`` is excluded: its ``--name`` names a tag being *created*.
# ``rename_tag`` (--old/--new) and ``merge_tags`` (--from/--into) name *two*
# tags via relational flags and are excluded by the same decision.
EXISTING_TAG_SCRIPTS = ["delete_tag", "assign_tag", "unassign_tag", "extract", "tag"]


def test_dispatcher_roster_derives_from_package():
    """The dispatcher's SCRIPTS is *derived* from the package, not hand-listed
    (#443). It must be exactly the non-underscore modules of
    ``bartleby.skill_scripts``, sorted — so a script dropped into that package
    is dispatchable (and convention-guarded above) with no parallel edit, while
    ``_``-prefixed shared modules stay excluded.
    """
    derived = sorted(
        m.name
        for m in pkgutil.iter_modules(skill_scripts.__path__)
        if not m.name.startswith("_")
    )
    assert list(SCRIPTS) == derived, (
        "the dispatcher's SCRIPTS must equal the sorted non-underscore modules "
        "of bartleby.skill_scripts; it is no longer a hand-maintained list"
    )
    assert SCRIPTS, "the dispatcher must advertise at least one script"
    assert not any(name.startswith("_") for name in SCRIPTS), (
        "underscore-prefixed helper modules must never appear in SCRIPTS"
    )

# Scripts that support corpus scoping — each must carry ``--in-documents`` (#408).
SCOPE_SCRIPTS = ["search", "scan", "describe_corpus", "list_documents"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_pagination_flags_use_shared_validators(script):
    """#403: ``--limit`` is ``positive_int``; ``--offset`` is ``nonneg_int``."""
    parser = _parser_for(script)
    limit = _action_for(parser, "--limit")
    if limit is not None:
        assert limit.type is positive_int, (
            f"{script} --limit must use the shared positive_int validator, "
            f"got type={getattr(limit.type, '__name__', limit.type)!r}"
        )
    offset = _action_for(parser, "--offset")
    if offset is not None:
        assert offset.type is nonneg_int, (
            f"{script} --offset must use the shared nonneg_int validator, "
            f"got type={getattr(offset.type, '__name__', offset.type)!r}"
        )


@pytest.mark.parametrize("script", EXISTING_TAG_SCRIPTS)
def test_existing_tag_flag_is_named_tag(script):
    """#405: the existing-tag selector is ``--tag``, never ``--name``/``--tag-name``."""
    options = _option_strings(_parser_for(script))
    assert "--tag" in options, (
        f"{script} identifies an existing tag and must name that flag --tag"
    )
    for misnamed in ("--name", "--tag-name"):
        assert misnamed not in options, (
            f"{script} uses {misnamed} for an existing tag; the convention is "
            "--tag (--name is reserved for the tag being created in add_tag)"
        )


@pytest.mark.parametrize("script", SCOPE_SCRIPTS)
def test_scope_scripts_expose_in_documents(script):
    """#408: every scope-supporting script carries ``--in-documents``."""
    options = _option_strings(_parser_for(script))
    assert "--in-documents" in options, (
        f"{script} supports corpus scoping and must expose --in-documents"
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_id_flag_suffix_matches_dest(script):
    """#573 (supersedes #111): a flag's option string ends in ``-id`` IFF its
    argparse ``dest`` ends in ``_id``.

    This is the *name the value you accept* convention as a biconditional, so it
    cannot drift back to the bare-noun #111 form (a renamed-away ``--finding``
    re-suffixes its dest, a freshly-added ``--document`` that takes an id trips
    the other arm). Plural/relational id flags (``dest`` ends ``_ids`` or another
    stem — ``--documents``/``--chunks``/``--from``/``--into``/``--around-chunk``)
    stay bare and pass both arms.
    """
    parser = _parser_for(script)
    mismatches = []
    for action in parser._actions:
        if not action.option_strings:
            continue  # positionals carry no convention here
        dest_is_id = action.dest.endswith("_id")
        for opt in action.option_strings:
            flag_is_id = opt.endswith("-id")
            if flag_is_id != dest_is_id:
                mismatches.append((opt, action.dest))
    assert not mismatches, (
        f"{script} violates the #573 id-flag biconditional (flag ends -id IFF "
        f"dest ends _id): {sorted(mismatches)}. An id-valued flag must end in "
        "-id (--finding-id, --document-id, --chunk-id); a name/key/path flag "
        "stays bare. Plural id lists keep an _ids/other-stem dest and stay bare."
    )


# ---- #573 regression grep: no stale bare id-flag invocation survives ------
#
# The biconditional above guards the *live* argparse parsers; this guards every
# other place an old flag spelling could regress — docs, examples, --help/epilog
# prose, tests, fixtures. It matches an actual flag *invocation* (the token
# ``--finding`` / ``--document`` / ``--chunk`` bounded by non-word, non-hyphen on
# both sides, so ``--finding-id`` / ``--findings`` / ``--documents`` /
# ``--around-chunk`` / the ``surface--finding`` CSS class never trip it).
#
# Two paths are allow-listed because they *legitimately* name the old spelling:
#   - ``docs/decisions/`` — the GH-0573 record documents the old→new rename, and
#     prior decision records quote the historical bare-noun convention as the
#     "why" of past calls (they must not be rewritten).
#   - this test file — its docstrings spell out both the old and new forms.
# (There is no in-repo CHANGELOG; release notes live in GitHub Releases, off VC.)

_STALE_FLAG_RE = re.compile(r"(?<![\w-])--(?:finding|document|chunk)(?![\w-])")

_GREP_ALLOWLIST = (
    "docs/decisions/",
    "tests/test_skill_flag_conventions.py",
)


def _version_controlled_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def test_no_stale_bare_id_flag_anywhere():
    """#573: no old bare id-flag invocation (``--finding`` / ``--document`` /
    ``--chunk``) survives anywhere under version control, except the allow-listed
    decision records and this test's own docstrings."""
    offenders: dict[str, list[int]] = {}
    for rel in _version_controlled_files():
        # Allow-list entries ending in "/" are dir prefixes; bare paths are
        # exact files — hence the ``==`` *or* ``startswith`` pair.
        if any(rel == p or rel.startswith(p) for p in _GREP_ALLOWLIST):
            continue
        path = _REPO_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary/asset or removed-but-staged; nothing to scan
        hits = [
            i for i, line in enumerate(text.splitlines(), 1)
            if _STALE_FLAG_RE.search(line)
        ]
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "stale bare id-flag invocation(s) survive the #573 rename — rename to "
        f"--finding-id / --document-id / --chunk-id:\n{offenders}"
    )
