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
  - #111 — no flag carries a redundant ``-id`` suffix (the convention is
    ``--document`` / ``--finding`` / ``--chunk``, with ``dest`` carrying the
    ``_id``, not the option string).

By design these assertions are LITERAL: each names the exact scripts and exact
flags the conventions cover. They are a tripwire, not a re-derivation of the
spec — no plural/arity/comma-list generalization that would turn the guard into
a brittle mirror of every script's full signature.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil

import pytest

from bartleby import skill_scripts
from bartleby.commands.skill import SCRIPTS
from bartleby.skill_scripts._common import nonneg_int, positive_int


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
def test_no_redundant_id_suffixed_flags(script):
    """#111: no flag ends in ``-id`` (use --document/--finding/--chunk; the
    ``_id`` lives on ``dest``, not the option string)."""
    offenders = sorted(
        opt for opt in _option_strings(_parser_for(script)) if opt.endswith("-id")
    )
    assert not offenders, (
        f"{script} has redundant -id-suffixed flag(s) {offenders}; name them "
        "--document / --finding / --chunk (the _id belongs on dest only)"
    )
