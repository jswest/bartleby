"""Shared helpers for tag CRUD, similarity, and classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from bartleby.config import ensure_provider_env, load_config
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
    value_type: str | None = None
    pattern: str | None = None

    @property
    def is_value_tag(self) -> bool:
        """A value-tag carries a method (value_type set); else a boolean tag."""
        return self.value_type is not None


def fetch_vocabulary(conn) -> list[TagRow]:
    rows = conn.cursor().execute(
        "SELECT tag_id, name, description, value_type, pattern "
        "FROM tags ORDER BY name"
    ).fetchall()
    return [TagRow(*row) for row in rows]


def get_tag_by_name(conn, name: str) -> TagRow | None:
    row = conn.cursor().execute(
        "SELECT tag_id, name, description, value_type, pattern "
        "FROM tags WHERE name = ?",
        (name,),
    ).fetchone()
    return TagRow(*row) if row else None


def find_tag_by_normalized_name(conn, name: str) -> TagRow | None:
    """Return the first tag whose normalized name equals ``normalize_name(name)``, else None.

    Normalized-equality check only — no embedding/similarity leg.
    """
    target_norm = normalize_name(name)
    for tag in fetch_vocabulary(conn):
        if normalize_name(tag.name) == target_norm:
            return tag
    return None


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

    Order is preserved across the input list. The named query unit behind
    ``resolve_scope``'s ``--tag`` intersection.
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
    ``echo_into``.

    ``temp_table`` is set when ``--file-like`` produced a scope that was
    materialized into a SQLite temp table instead of a Python list (to avoid
    binding >32 k variables). When set, ``document_ids`` is ``None`` (the
    Python list is not populated) and callers must use ``restrict_in`` rather
    than reading ``document_ids`` directly for SQL predicate building.
    """

    document_ids: list[int] | None
    in_documents: list[int] | None      # echo: as requested (pre-resolution)
    tags: list[str] | None              # echo: requested tag names
    file_like: list[str] | None         # echo: requested LIKE patterns (OR group)
    authored_after: str | None
    authored_before: str | None
    include_nulls: bool
    excluded_null_dated: int
    temp_table: str | None = None       # set when file-like scope lives in a temp table

    @property
    def date_active(self) -> bool:
        return self.authored_after is not None or self.authored_before is not None

    @property
    def active(self) -> bool:
        """True if any scope filter was requested (tags, in_documents, file_like, or a date bound)."""
        return (
            self.in_documents is not None
            or self.tags is not None
            or self.file_like is not None
            or self.date_active
        )

    def echo_into(self, env: dict) -> dict:
        """Attach the self-describing ``filters`` echo to ``env`` in place when
        scoped; return it.

        Folds the "set the ``filters`` key only when a filter is active" dance
        that ``search``, ``scan``, ``describe_corpus``, and ``list_documents``
        would otherwise each repeat.
        """
        if self.active:
            env["filters"] = {
                "tags": self.tags,
                "in_documents": self.in_documents,
                "file_like": self.file_like,
                "authored_after": self.authored_after,
                "authored_before": self.authored_before,
                "include_nulls": self.include_nulls,
                "excluded_null_dated": self.excluded_null_dated,
            }
        return env

    def restrict_in(self, col: str) -> tuple[str, list]:
        """A bare ``document_ids`` predicate for ``col`` plus its params.

        Returns one of four forms, *without* a leading ``WHERE``/``AND`` so the
        caller composes it freely:
          - ``("", [])`` — whole corpus, no restriction.
          - ``("0", [])`` — a filter matched nothing; matches no rows.
          - ``("<col> IN (?,?)", [ids...])`` — restrict to a small resolved slice.
          - ``("<col> IN (SELECT document_id FROM <temp>)", [])`` — restrict via
            temp-table join when ``--file-like`` produced a large match set.
        """
        if self.temp_table is not None:
            return f"{col} IN (SELECT document_id FROM {self.temp_table})", []
        if self.document_ids is None:
            return "", []
        if not self.document_ids:
            return "0", []
        ph = ",".join("?" * len(self.document_ids))
        return f"{col} IN ({ph})", list(self.document_ids)


def _date_bounds(after: str | None, before: str | None) -> tuple[str, list]:
    """``("s.authored_date >= ? AND …", [params])`` for an active date bound."""
    clauses, params = [], []
    if after is not None:
        clauses.append("s.authored_date >= ?")
        params.append(after)
    if before is not None:
        clauses.append("s.authored_date <= ?")
        params.append(before)
    return " AND ".join(clauses), params


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
    bounds_sql, bound_params = _date_bounds(after, before)

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


def _apply_date_bound_to_temp(
    conn, temp_table: str, *,
    after: str | None, before: str | None, include_nulls: bool,
) -> int:
    """Apply a date bound by deleting non-qualifying rows from ``temp_table`` in place.

    Used when the scope was materialized into a temp table (the file-like path).
    Removes rows that don't satisfy the date bound, returning ``excluded_null_dated``.
    The temp table is modified in place so callers retain the same table name.
    """
    bounds_sql, bound_params = _date_bounds(after, before)

    cur = conn.cursor()
    excluded = 0

    if include_nulls:
        # Keep rows where date is NULL OR satisfies the bound; drop the rest.
        cur.execute(
            f"DELETE FROM {temp_table} WHERE document_id NOT IN ("
            f"  SELECT t.document_id FROM {temp_table} t "
            f"  LEFT JOIN summaries s USING (document_id) "
            f"  WHERE s.authored_date IS NULL OR ({bounds_sql})"
            f")",
            bound_params,
        )
    else:
        # Count undated rows (excluded from any date-bound result), then remove
        # all rows that don't satisfy the bound. The INNER JOIN naturally drops
        # undated rows (no summaries row → no join hit), so no separate null pass.
        excluded = cur.execute(
            f"SELECT COUNT(*) FROM {temp_table} t "
            f"LEFT JOIN summaries s USING (document_id) "
            f"WHERE s.authored_date IS NULL",
        ).fetchone()[0]
        cur.execute(
            f"DELETE FROM {temp_table} WHERE document_id NOT IN ("
            f"  SELECT t.document_id FROM {temp_table} t "
            f"  JOIN summaries s USING (document_id) "
            f"  WHERE {bounds_sql}"
            f")",
            bound_params,
        )
    return excluded


def resolve_scope(
    conn, *,
    in_documents: list[int] | None = None,
    tags: list[str] | None = None,
    file_like: list[str] | None = None,
    authored_after: str | None = None,
    authored_before: str | None = None,
    include_nulls: bool = False,
) -> Scope:
    """Fold ``--tag`` ∩ ``--in-documents`` ∩ ``--file-like`` ∩ date bounds into one ``Scope``.

    The single scope resolver for ``list_documents``, ``scan``, ``search``, and
    ``describe_corpus``. Tags and ``in_documents`` intersect; ``--file-like``
    (LIKE patterns OR'd together) then intersects that set; an active date bound
    finally narrows the result (and accounts for the undated docs it drops).
    Each intersection yields ``[]`` (short-circuit to zero hits) when empty, or
    passes the prior scope through unchanged when its filter is absent.
    Unknown tags raise ``TAG_NOT_FOUND`` and malformed bounds raise ``INVALID_DATE``.

    When ``--file-like`` is present the matched ids are materialized into a
    SQLite temp table (``_scope_file_like``) instead of a Python list, so a
    broad glob that matches hundreds of thousands of documents never triggers
    SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` limit. The returned ``Scope`` carries
    ``temp_table`` set to the table name; ``restrict_in`` returns a subquery
    predicate against it rather than an ``IN (?, ?, …)`` list.
    """
    after = validate_date_bound("--authored-after", authored_after)
    before = validate_date_bound("--authored-before", authored_before)

    date_active = after is not None or before is not None

    def _empty_scope(excluded: int = 0) -> "Scope":
        """A zero-hit scope carrying the active filters (only the dropped-NULL
        count varies between the short-circuit points)."""
        return Scope(
            document_ids=[],
            in_documents=in_documents,
            tags=tags or None,
            file_like=file_like,
            authored_after=after,
            authored_before=before,
            include_nulls=include_nulls,
            excluded_null_dated=excluded,
        )

    if file_like:
        # Build the temp table: start with all file-like matches, then intersect
        # with tags and in_documents inside SQLite (never in Python) to avoid
        # a high-cardinality bind list at every consumer.
        tag_ids: list[int] = []
        if tags:
            tag_ids = resolve_tag_names(conn, tags)

        like_clause = " OR ".join("d.file_name LIKE ?" for _ in file_like)
        conditions: list[str] = [f"({like_clause})"]
        params: list = list(file_like)

        if in_documents is not None:
            if not in_documents:
                # Explicit empty in_documents → nothing matches.
                return _empty_scope()
            id_ph = ",".join("?" * len(in_documents))
            conditions.append(f"d.document_id IN ({id_ph})")
            params.extend(in_documents)

        if tag_ids:
            tid_ph = ",".join("?" * len(tag_ids))
            conditions.append(
                f"EXISTS (SELECT 1 FROM document_tags dt "
                f"WHERE dt.document_id = d.document_id AND dt.tag_id IN ({tid_ph}))"
            )
            params.extend(tag_ids)

        where_sql = " AND ".join(conditions)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS temp._scope_file_like")
        cur.execute(
            "CREATE TEMP TABLE _scope_file_like (document_id INTEGER PRIMARY KEY)"
        )
        cur.execute(
            f"INSERT INTO _scope_file_like "
            f"SELECT DISTINCT d.document_id FROM documents d WHERE {where_sql}",
            params,
        )

        # Check whether the temp table is empty (whole filter matched nothing).
        count = cur.execute(
            "SELECT COUNT(*) FROM _scope_file_like"
        ).fetchone()[0]
        if count == 0:
            return _empty_scope()

        excluded = 0
        if date_active:
            excluded = _apply_date_bound_to_temp(
                conn, "_scope_file_like",
                after=after, before=before, include_nulls=include_nulls,
            )
            # After date filtering, check again for empty.
            count = cur.execute(
                "SELECT COUNT(*) FROM _scope_file_like"
            ).fetchone()[0]
            if count == 0:
                return _empty_scope(excluded)

        return Scope(
            document_ids=None,
            in_documents=in_documents,
            tags=tags or None,
            file_like=file_like,
            authored_after=after,
            authored_before=before,
            include_nulls=include_nulls,
            excluded_null_dated=excluded,
            temp_table="_scope_file_like",
        )

    # No file-like: use the original Python-list path.
    scoped_docs = in_documents
    if tags:
        tagged = documents_with_any_tag(conn, resolve_tag_names(conn, tags))
        scoped_docs = tagged if scoped_docs is None else sorted(set(scoped_docs) & set(tagged))

    if not date_active:
        document_ids, excluded = scoped_docs, 0
    elif scoped_docs is not None and not scoped_docs:
        # The tag/in-documents scope is already empty — the bound can't add anything.
        document_ids, excluded = [], 0
    else:
        document_ids, excluded = _apply_date_bound(
            conn, scoped_docs,
            after=after, before=before, include_nulls=include_nulls,
        )

    return Scope(
        document_ids=document_ids,
        in_documents=in_documents,
        tags=tags or None,
        file_like=file_like,
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

    ``embed_texts`` is imported lazily (not at module top) so the FTS-only read
    scripts that share this module via ``resolve_scope`` — ``scan``,
    ``list_documents``, ``read_chunks``, ``describe_corpus`` — never pay the
    embedding/model stack's import cost (#371).
    """
    vocab = fetch_vocabulary(conn)
    if not vocab:
        return None

    target_norm = normalize_name(name)
    for tag in vocab:
        if normalize_name(tag.name) == target_norm:
            return SimilarTag(tag.tag_id, tag.name, tag.description, 1.0)

    from bartleby.ingest.embed import embed_texts

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


# ---------- value-bearing tags ----------


# The three casts a value-tag may declare. ``value_type`` is also the
# discriminator: NULL = an ordinary boolean tag, non-NULL = a value-tag.
VALUE_TYPES = ("number", "string", "date")

# A value-tag's pattern MUST expose the captured substring through a named
# group ``(?P<value>…)`` so extraction is unambiguous about which span to
# store. (re2 supports named groups identically to stock ``re``.)
_VALUE_GROUP = "value"
# Minimal ISO-date shape we require when ``value_type == "date"``: YYYY-MM-DD.
# The pattern isolates a date substring; the cast just validates its shape so
# stored dates sort and compare like the summarizer's ``authored_date``.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def compile_pattern(pattern: str):
    """Compile a value-tag pattern with re2, requiring a ``(?P<value>…)`` group.

    re2 (google-re2) guarantees linear-time matching, so agent-authored
    patterns run across many chunks can't trigger catastrophic backtracking
    the way stock ``re`` can. Raises ``INVALID_PATTERN`` on a malformed regex
    or a pattern missing the named ``value`` capture group.
    """
    import re2

    try:
        compiled = re2.compile(pattern)
    except Exception as e:  # re2 raises its own error types on bad syntax
        raise SkillError(
            "INVALID_PATTERN", f"Pattern is not a valid regex: {e}",
        ) from None
    if _VALUE_GROUP not in (compiled.groupindex or {}):
        raise SkillError(
            "INVALID_PATTERN",
            "Pattern must contain a named capture group "
            "(?P<value>…) marking the substring to extract.",
        )
    return compiled


def normalize_number(raw: str) -> str:
    """Strip ``$ , %`` and read ``(…)`` as negative, then validate as a number.

    The regex only has to *isolate* a numeric substring; this normalizer cleans
    common accounting formatting before the cast so patterns stay simple. The
    canonical stored form is the cleaned numeric string (e.g. ``"(1,234.5)"`` →
    ``"-1234.5"``). Raises ``INVALID_VALUE`` if the result isn't a number.
    """
    s = raw.strip()
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    if negative and s and not s.startswith("-"):
        s = "-" + s
    try:
        float(s)
    except ValueError:
        raise SkillError(
            "INVALID_VALUE",
            f"Captured {raw!r} does not normalize to a number.",
        ) from None
    return s


def cast_value(value_type: str, captured: str) -> str:
    """Cast a captured substring per ``value_type``, returning canonical text.

    - ``number``: normalized via :func:`normalize_number`.
    - ``date``: must already be ``YYYY-MM-DD`` (the pattern isolates it).
    - ``string``: stored verbatim (stripped).

    Stored canonically as text in every case (cast on read by the consumer).
    Raises ``INVALID_VALUE`` when the captured span can't satisfy the type.
    """
    if value_type == "number":
        return normalize_number(captured)
    if value_type == "date":
        s = captured.strip()
        if not _DATE_RE.match(s):
            raise SkillError(
                "INVALID_VALUE",
                f"Captured {captured!r} is not a YYYY-MM-DD date.",
            )
        return s
    return captured.strip()


def require_value_tag(tag: TagRow) -> None:
    """Raise ``NOT_A_VALUE_TAG`` unless ``tag`` carries a value method."""
    if not tag.is_value_tag:
        raise SkillError(
            "NOT_A_VALUE_TAG",
            f"Tag {tag.name!r} is a boolean tag (no value_type/pattern); "
            "create it with --value-type/--pattern to extract values.",
        )


def upsert_value(
    conn, document_id: int, tag_id: int, value: str, chunk_id: int | None,
) -> None:
    """Write a document's value for a tag, anchored to ``chunk_id``.

    One value per ``(tag, document)`` by the table's PK: an existing row is
    overwritten (ON CONFLICT), so a re-extraction replaces the prior value and
    its anchor rather than erroring or duplicating.
    """
    conn.cursor().execute(
        "INSERT INTO document_tags (document_id, tag_id, value, chunk_id) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (document_id, tag_id) DO UPDATE SET "
        "value = excluded.value, chunk_id = excluded.chunk_id",
        (document_id, tag_id, value, chunk_id),
    )
