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


def require_tag_by_name(conn, name: str) -> TagRow:
    """``get_tag_by_name`` that raises ``TAG_NOT_FOUND`` instead of returning None.

    The lookup-or-raise every tag command does on its target tag (assign/
    unassign/delete/rename/merge/tag); folds the repeated ``if tag is None:
    raise`` into one place."""
    tag = get_tag_by_name(conn, name)
    if tag is None:
        raise SkillError("TAG_NOT_FOUND", f"No tag named {name!r}.")
    return tag


def resolve_documents(
    conn, document_ids: list[int],
) -> tuple[list[tuple[int, str]], list[int]]:
    """Split requested ids into ``(found, not_found)`` in one query.

    ``found`` is ``[(document_id, file_name), ...]`` and ``not_found`` is the
    ids with no document — both in first-occurrence input order, deduped. Lets
    a batch assign/unassign skip absent ids without aborting the rest (the
    per-document analogue of ``tag``'s ``failed`` list).
    """
    ordered = list(dict.fromkeys(document_ids))  # dedup, preserve order
    placeholders = ",".join("?" * len(ordered))
    names = dict(conn.cursor().execute(
        f"SELECT document_id, file_name FROM documents "
        f"WHERE document_id IN ({placeholders})",
        ordered,
    ))
    found = [(d, names[d]) for d in ordered if d in names]
    not_found = [d for d in ordered if d not in names]
    return found, not_found


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


def intersect_tag_filter(
    conn, in_documents: list[int] | None, tag_names: list[str] | None,
) -> tuple[list[int] | None, list[str] | None]:
    """Fold ``--tag`` into ``in_documents`` as an intersection.

    Shared by ``search`` and ``scan``. Without tags, ``in_documents`` passes
    through unchanged. With tags, the result is the intersection of the
    explicit document set (if any) and the documents carrying any of the named
    tags. An empty intersection yields ``[]`` — the caller short-circuits to
    zero hits.
    """
    if not tag_names:
        return in_documents, None
    tagged = documents_with_any_tag(conn, resolve_tag_names(conn, tag_names))
    if in_documents is None:
        return tagged, tag_names
    return sorted(set(in_documents) & set(tagged)), tag_names


# ---------- scope resolution (tags + in_documents + date bounds) ----------


def validate_date_bound(flag: str, raw: str | None) -> str | None:
    """Validate a user-supplied date *bound*, raising ``INVALID_DATE`` on junk.

    ``normalize_authored_date`` returns None on unparseable input, which is
    right for summarizer-inferred *storage* but a footgun for a *filter*: a
    silently-dropped bound would widen the result without warning. So we reuse
    its parsing but raise instead of nulling.
    """
    from bartleby.ingest.summarize import normalize_authored_date

    if raw is None:
        return None
    norm = normalize_authored_date(raw)
    if norm is None:
        raise SkillError(
            "INVALID_DATE",
            f"{flag} must be a real calendar date in YYYY-MM-DD form; got {raw!r}.",
        )
    return norm


@dataclass
class Scope:
    """A resolved corpus scope: which documents, plus the echo of how it was asked.

    ``document_ids`` is the concrete set the caller restricts to — ``None``
    means "whole corpus" (no filter), ``[]`` means "a filter was applied but it
    matched nothing" (short-circuit to zero results). The remaining fields echo
    the *requested* filter so a response can be self-describing via
    ``filters_dict``.
    """

    document_ids: list[int] | None
    in_documents: list[int] | None      # echo: as requested (pre-resolution)
    tags: list[str] | None              # echo: requested tag names
    authored_after: str | None
    authored_before: str | None
    include_nulls: bool
    excluded_null_dated: int

    @property
    def date_active(self) -> bool:
        return self.authored_after is not None or self.authored_before is not None

    @property
    def active(self) -> bool:
        """True if any scope filter was requested (tags, in_documents, or a date bound)."""
        return (
            self.in_documents is not None
            or self.tags is not None
            or self.date_active
        )

    def filters_dict(self) -> dict | None:
        """The self-describing ``filters`` echo, or ``None`` when unfiltered.

        Shared by ``scan`` and ``describe_corpus`` so both surface scope
        identically. ``list_documents`` predates this and keeps its own
        top-level shape (the issue mandates no behavior change there).
        """
        if not self.active:
            return None
        return {
            "tags": self.tags,
            "in_documents": self.in_documents,
            "authored_after": self.authored_after,
            "authored_before": self.authored_before,
            "include_nulls": self.include_nulls,
            "excluded_null_dated": self.excluded_null_dated,
        }

    def echo_into(self, env: dict) -> dict:
        """Attach the ``filters`` echo to ``env`` in place when scoped; return it.

        Folds the "compute filters_dict, set the key only if non-None" dance that
        ``scan`` and ``describe_corpus`` would otherwise each repeat.
        """
        filters = self.filters_dict()
        if filters is not None:
            env["filters"] = filters
        return env

    def restrict_in(self, col: str) -> tuple[str, list]:
        """A bare ``document_ids`` predicate for ``col`` plus its params.

        Returns one of three forms, *without* a leading ``WHERE``/``AND`` so the
        caller composes it freely:
          - ``("", [])`` — whole corpus, no restriction.
          - ``("0", [])`` — a filter matched nothing; matches no rows.
          - ``("<col> IN (?,?)", [ids...])`` — restrict to the resolved slice.
        This is the single source for the None/empty/list branch all three
        scripts otherwise hand-roll.
        """
        if self.document_ids is None:
            return "", []
        if not self.document_ids:
            return "0", []
        ph = ",".join("?" * len(self.document_ids))
        return f"{col} IN ({ph})", list(self.document_ids)


def _apply_date_bound(
    conn, base_ids: list[int] | None, *,
    after: str | None, before: str | None, include_nulls: bool,
) -> tuple[list[int], int]:
    """Resolve an active date bound to ``(document_ids, excluded_null_dated)``.

    ``base_ids`` is the tag/in-documents scope the bound layers on top of —
    ``None`` means the whole corpus, otherwise a non-empty list (callers
    short-circuit an empty base before getting here). Mirrors the semantics
    ``list_documents`` shipped: undated docs can't satisfy a bound, so they're
    excluded by default and counted in ``excluded_null_dated``; ``include_nulls``
    keeps them and zeros the count.
    """
    bounds, bound_params = [], []
    if after is not None:
        bounds.append("s.authored_date >= ?")
        bound_params.append(after)
    if before is not None:
        bounds.append("s.authored_date <= ?")
        bound_params.append(before)
    bounds_sql = " AND ".join(bounds)

    base_sql, base_params = "", []
    if base_ids is not None:
        ph = ",".join("?" * len(base_ids))
        base_sql = f" AND d.document_id IN ({ph})"
        base_params = list(base_ids)

    if include_nulls:
        date_pred = f"(s.authored_date IS NULL OR ({bounds_sql}))"
    else:
        date_pred = f"(s.authored_date IS NOT NULL AND {bounds_sql})"

    cur = conn.cursor()
    document_ids = sorted(
        row[0] for row in cur.execute(
            f"SELECT d.document_id FROM documents d "
            f"LEFT JOIN summaries s USING (document_id) "
            f"WHERE {date_pred}{base_sql}",
            [*bound_params, *base_params],
        )
    )

    excluded = 0
    if not include_nulls:
        excluded = cur.execute(
            f"SELECT COUNT(*) FROM documents d "
            f"LEFT JOIN summaries s USING (document_id) "
            f"WHERE s.authored_date IS NULL{base_sql}",
            base_params,
        ).fetchone()[0]
    return document_ids, excluded


def resolve_scope(
    conn, *,
    in_documents: list[int] | None = None,
    tags: list[str] | None = None,
    authored_after: str | None = None,
    authored_before: str | None = None,
    include_nulls: bool = False,
) -> Scope:
    """Fold ``--tag`` ∩ ``--in-documents`` ∩ date bounds into one ``Scope``.

    The single scope resolver for ``list_documents``, ``scan``, and
    ``describe_corpus``. Tags and ``in_documents`` intersect via
    ``intersect_tag_filter``; an active date bound then narrows that set (and
    accounts for the undated docs it drops). Unknown tags raise ``TAG_NOT_FOUND``
    and malformed bounds raise ``INVALID_DATE``.
    """
    after = validate_date_bound("--authored-after", authored_after)
    before = validate_date_bound("--authored-before", authored_before)

    scoped_docs, tag_names = intersect_tag_filter(conn, in_documents, tags)
    date_active = after is not None or before is not None

    if not date_active:
        document_ids, excluded = scoped_docs, 0
    elif scoped_docs is not None and not scoped_docs:
        # Tag/in-documents scope already empty — the bound can't add anything.
        document_ids, excluded = [], 0
    else:
        document_ids, excluded = _apply_date_bound(
            conn, scoped_docs,
            after=after, before=before, include_nulls=include_nulls,
        )

    return Scope(
        document_ids=document_ids,
        in_documents=in_documents,
        tags=tag_names,
        authored_after=after,
        authored_before=before,
        include_nulls=include_nulls,
        excluded_null_dated=excluded,
    )


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
            "No LLM provider configured. Run `bartleby config` first.",
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
