"""Type-tagged entity ids for the agent-facing interface (issue #624).

Every id a skill script emits to stdout JSON is rendered ``<type>:<id>`` (e.g.
``chunk:15837``, ``document:204``), and every ``--*-id``-style flag and the
``[^chunk:N]`` citation marker accepts *only* the prefixed form. The point is to
make chunk-vs-document confusion structurally impossible: ``chunk_id`` and
``document_id`` are independent INTEGER PKs whose integer ranges overlap, so a
bare int is ambiguous and was silently mis-stored as a citation in real
sessions (#623).

This is a pure interface layer — parse-on-input, format-on-output. Storage stays
integer (``finding_citations`` rows are bare ints); there is no schema change.
The cutover is hard: bare ints on input are *rejected*, never silently accepted.
"""

from __future__ import annotations

import argparse

# The fixed field-name → entity-type map for output formatting. ``source_id`` is
# deliberately absent — its real type depends on the row's ``source_kind``, so it
# is prefixed by kind explicitly at its emission sites, never through this map.
_OUTPUT_FIELD_TYPES = {
    "chunk_id": "chunk",
    "chunk_ids": "chunk",
    "dangling_citations": "chunk",       # [^chunk:N] markers whose citation no longer resolves
    "dangling_finding_links": "finding", # [^finding:N] markers whose target no longer exists (#654)
    "document_id": "document",
    # The ``filters`` scope echo (search/scan/list_documents/describe_corpus)
    # carries the as-requested document ids under ``in_documents``.
    "in_documents": "document",
    "finding_id": "finding",
    "image_id": "image",
    "tag_id": "tag",
    # tag's full-vocab verdict lists the assigned tag ids per document.
    "assigned_tag_ids": "tag",
}


def format_id(id_type: str, value: int | None) -> str | None:
    """Render ``value`` as ``"<id_type>:<value>"`` (``None`` passes through)."""
    if value is None:
        return None
    return f"{id_type}:{value}"


def format_source_id(source_kind: str, source_id: int | None) -> str | None:
    """Prefix a chunk's ``source_id`` by its ``source_kind``.

    A chunk's ``source_id`` is polymorphic — its real type is the row's
    ``source_kind`` (``document`` / ``summary`` / ``finding`` / ``image``), which
    map one-to-one to id types. This can't go through the flat field map (the
    type isn't knowable from the key alone), so emission sites call it directly.
    """
    if source_id is None:
        return None
    return format_id(source_kind, source_id)


def parse_id(value: str, expected_type: str) -> int:
    """Parse a ``"<expected_type>:<int>"`` token into its bare int.

    Accepts *only* the prefixed form: a bare int (the pre-#624 shape) is
    rejected, as is a correctly-shaped value carrying the wrong type (e.g. a
    ``document:`` value handed to a chunk flag). Raises
    :class:`argparse.ArgumentTypeError`.
    """
    text = value.strip()
    # Distinguish "bare int" from "wrong type" so the message points at the fix.
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be a type-tagged id of the form {expected_type}:<int> "
            f"(e.g. {expected_type}:204); bare ids are no longer accepted"
        )
    other, body = text.split(":", 1)
    if other != expected_type:
        raise argparse.ArgumentTypeError(
            f"'{value}' is a {other} id, but this flag takes a {expected_type} id "
            f"({expected_type}:<int>)"
        )
    try:
        n = int(body)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid {expected_type} id "
            f"(expected {expected_type}:<int>)"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"{expected_type} id must be a positive integer; got {value!r}"
        )
    return n


def prefixed_int(expected_type: str):
    """argparse ``type=`` for a single ``"<expected_type>:<int>"`` id."""
    def _parse(value: str) -> int:
        return parse_id(value, expected_type)
    return _parse


def prefixed_int_list(expected_type: str):
    """argparse ``type=`` for a comma-separated list of typed ids.

    Every element must carry ``expected_type:``; a bare int or wrong-type
    element fails the whole list (loudly, not silently looking up a colliding
    row). At least one element is required.
    """
    def _parse(value: str) -> list[int]:
        out: list[int] = []
        for piece in value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            out.append(parse_id(piece, expected_type))
        if not out:
            raise argparse.ArgumentTypeError(
                f"at least one {expected_type} id required "
                f"(form {expected_type}:<int>)"
            )
        return out
    return _parse


def format_output_ids(obj):
    """Recursively prefix every known id field in a dict/list output structure.

    Walks ``obj`` (a dict, list, or scalar) and, for any dict key in the fixed
    field→type map, rewrites its value to the ``"<type>:<id>"`` form — handling
    a scalar id, a list of ids, and ``None`` (left as ``None``). Keys not in the
    map are recursed into but otherwise untouched. ``source_id`` is *not* in the
    map; format it by ``source_kind`` at its emission site.
    """
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            id_type = _OUTPUT_FIELD_TYPES.get(key)
            if id_type is not None:
                out[key] = _format_field(id_type, val)
            else:
                out[key] = format_output_ids(val)
        return out
    if isinstance(obj, list):
        return [format_output_ids(item) for item in obj]
    return obj


def _format_field(id_type: str, val):
    """Prefix a single id field's value: scalar int, list of ints, or None."""
    if val is None:
        return None
    if isinstance(val, list):
        return [format_id(id_type, v) for v in val]
    return format_id(id_type, val)
