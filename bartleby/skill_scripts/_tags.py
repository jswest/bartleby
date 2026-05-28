"""Shared helpers for tag CRUD, similarity, and classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from bartleby.config import ensure_provider_env, load_config
from bartleby.ingest.embed import embed_texts
from bartleby.providers import Provider, get_provider
from bartleby.skill_runner import SkillError


# Above this cosine similarity (against an existing tag's description) we
# treat a proposed tag as a likely duplicate. 0.85 is the issue's published
# default; tune by edit later if the corpus needs it.
SIMILARITY_THRESHOLD = 0.85


# ---------- structured-output schemas for the classifier ----------


class _TagsAssignment(BaseModel):
    tag_ids: list[int] = Field(
        description=(
            "The subset of tag_ids from the provided vocabulary that apply "
            "to this document. Empty list if none apply."
        ),
    )


class _TagApplies(BaseModel):
    applies: bool = Field(
        description=(
            "True if the single tag applies to the document, false otherwise."
        ),
    )


# ---------- name normalization ----------


_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """Lowercase + strip non-alphanumerics for duplicate detection.

    ``"NYSEG"`` and ``"nyseg"`` and ``"ny-seg"`` all collapse to ``"nyseg"``.
    Used for the exact-match leg of conflict detection.
    """
    return _NAME_NORMALIZE_RE.sub("", name.lower())


# ---------- vocabulary I/O ----------


@dataclass
class TagRow:
    tag_id: int
    name: str
    description: str


def fetch_vocabulary(conn) -> list[TagRow]:
    rows = conn.cursor().execute(
        "SELECT tag_id, name, description FROM tags ORDER BY name"
    ).fetchall()
    return [TagRow(*row) for row in rows]


def get_tag_by_name(conn, name: str) -> TagRow | None:
    row = conn.cursor().execute(
        "SELECT tag_id, name, description FROM tags WHERE name = ?",
        (name,),
    ).fetchone()
    return TagRow(*row) if row else None


def get_document(conn, document_id: int) -> tuple[int, str] | None:
    """Return ``(document_id, file_name)`` for a document, or None if absent."""
    return conn.cursor().execute(
        "SELECT document_id, file_name FROM documents WHERE document_id = ?",
        (document_id,),
    ).fetchone()


def resolve_tag_names(conn, names: list[str]) -> list[int]:
    """Resolve tag names → tag_ids; raise ``TAG_NOT_FOUND`` on any miss.

    Order is preserved across the input list. Used by `list_documents`,
    `search`, and any future consumer of `--tag`.
    """
    placeholders = ",".join("?" * len(names))
    rows = conn.cursor().execute(
        f"SELECT name, tag_id FROM tags WHERE name IN ({placeholders})",
        names,
    ).fetchall()
    by_name = {n: tid for n, tid in rows}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SkillError(
            "TAG_NOT_FOUND",
            f"Unknown tag(s): {', '.join(repr(m) for m in missing)}.",
        )
    return [by_name[n] for n in names]


def documents_with_any_tag(conn, tag_ids: list[int]) -> list[int]:
    """Distinct document_ids carrying any of ``tag_ids`` (OR semantics)."""
    if not tag_ids:
        return []
    placeholders = ",".join("?" * len(tag_ids))
    return [
        row[0] for row in conn.cursor().execute(
            f"SELECT DISTINCT document_id FROM document_tags "
            f"WHERE tag_id IN ({placeholders})",
            tag_ids,
        )
    ]


# ---------- similarity check ----------


@dataclass
class SimilarTag:
    tag_id: int
    name: str
    description: str
    similarity: float


def find_similar_tag(
    conn, *, name: str, description: str,
) -> SimilarTag | None:
    """Return a probable-duplicate existing tag, or None.

    Two legs run independently and the higher-confidence hit wins:
      - normalized-name exact match — catches ``"NYSEG"`` vs ``"nyseg"``.
      - cosine of BGE embeddings against each existing description, with
        the configured ``SIMILARITY_THRESHOLD``.

    Embeddings are L2-normalized at the embedder, so dot-product is cosine.
    """
    vocab = fetch_vocabulary(conn)
    if not vocab:
        return None

    target_norm = normalize_name(name)
    for tag in vocab:
        if normalize_name(tag.name) == target_norm:
            return SimilarTag(tag.tag_id, tag.name, tag.description, 1.0)

    proposed_emb, *existing_embs = embed_texts(
        [description] + [t.description for t in vocab]
    )
    best: SimilarTag | None = None
    for tag, emb in zip(vocab, existing_embs):
        sim = sum(a * b for a, b in zip(proposed_emb, emb))
        if sim >= SIMILARITY_THRESHOLD and (best is None or sim > best.similarity):
            best = SimilarTag(tag.tag_id, tag.name, tag.description, sim)
    return best


# ---------- classification ----------


def resolve_classifier() -> tuple[Provider, str, float]:
    """Return (provider, model, temperature) from the user's summarizer config.

    Per the issue, tag classification reuses the summarizer's provider/model
    — no new config knob. Raises ``SkillError`` if no provider is configured.
    """
    config = load_config()
    name = config.get("provider")
    model = config.get("model")
    if not name or not model:
        raise SkillError(
            "NO_PROVIDER",
            "No LLM provider configured. Run `bartleby ready` first.",
        )
    ensure_provider_env(name, config)
    provider = get_provider(name, ollama_base_url=config.get("ollama_base_url"))
    temperature = float(config.get("temperature", 0))
    return provider, model, temperature


def classify_full_vocabulary(
    provider: Provider, model: str, temperature: float,
    *, summary_text: str, vocabulary: list[TagRow],
) -> list[int]:
    """Pick the subset of ``vocabulary`` that applies to ``summary_text``."""
    if not vocabulary:
        return []
    lines = [f"  {t.tag_id}: {t.name} — {t.description}" for t in vocabulary]
    prompt = (
        "Classify the document below against this controlled tag vocabulary. "
        "Return only the tag_ids that genuinely apply; the empty list is a "
        "valid answer.\n\n"
        f"Vocabulary:\n" + "\n".join(lines) + "\n\n"
        f"Document summary:\n{summary_text}"
    )
    result = provider.classify(
        prompt, model=model, schema=_TagsAssignment, temperature=temperature,
    )
    valid_ids = {t.tag_id for t in vocabulary}
    return [tid for tid in result.tag_ids if tid in valid_ids]


def classify_single_tag(
    provider: Provider, model: str, temperature: float,
    *, summary_text: str, tag: TagRow,
) -> bool:
    """Answer whether ``tag`` applies to ``summary_text``."""
    prompt = (
        "Decide whether the single tag below applies to the document.\n\n"
        f"Tag: {tag.name}\n"
        f"Description: {tag.description}\n\n"
        f"Document summary:\n{summary_text}"
    )
    result = provider.classify(
        prompt, model=model, schema=_TagApplies, temperature=temperature,
    )
    return result.applies


# ---------- summary lookup ----------


def summary_for(conn, document_id: int) -> str | None:
    """Return a document's summary text, or None if it has no summary.

    Classification feeds on the summary, not the body — summaries are 200–500
    tokens, cheap to embed and sufficient for document-level themes.
    """
    row = conn.cursor().execute(
        "SELECT text FROM summaries WHERE document_id = ?",
        (document_id,),
    ).fetchone()
    return row[0] if row else None


# ---------- assignment helpers ----------


def assign(conn, document_id: int, tag_ids: list[int]) -> None:
    """Insert one ``document_tags`` row per tag (OR IGNORE on duplicates)."""
    conn.cursor().executemany(
        "INSERT OR IGNORE INTO document_tags (document_id, tag_id) "
        "VALUES (?, ?)",
        [(document_id, tid) for tid in tag_ids],
    )


def unassign(conn, document_id: int, tag_id: int) -> None:
    """Drop one ``(document_id, tag_id)`` row (no-op if absent)."""
    conn.cursor().execute(
        "DELETE FROM document_tags WHERE document_id = ? AND tag_id = ?",
        (document_id, tag_id),
    )
