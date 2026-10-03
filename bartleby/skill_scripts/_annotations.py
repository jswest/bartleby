"""Shared JSON shape for an annotation in skill output (#689).

Underscore-prefixed so the skill dispatcher never exposes it as a command.
"""

from __future__ import annotations


def annotation_json(a: dict) -> dict:
    """One ``db.annotations`` row dict -> its skill-output shape (ids untagged)."""
    return {
        "annotation_id": a["annotation_id"],
        "body": a["body"],
        "anchor": None if a["anchor_exact"] is None else {
            "exact": a["anchor_exact"],
            "prefix": a["anchor_prefix"],
            "suffix": a["anchor_suffix"],
        },
        "anchor_found": a["anchor_found"],
        "chunk_id": a["chunk_id"],
        "is_human_author": bool(a["is_human_author"]),
        "session_id": a["session_id"],
        "created_at": a["created_at"],
    }
