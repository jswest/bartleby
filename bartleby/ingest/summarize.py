"""Document summarizer.

One whole-document summary per document, structured-output enforced across
all providers. Long documents are truncated to ``max_summarize_tokens`` of
the embedder-agnostic ``tiktoken cl100k_base`` encoder; the resulting
summary gets a deterministic truncation note appended in code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from bartleby.providers import DocumentSummary, Provider


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_authored_date(raw: str | None) -> str | None:
    """Accept only strict YYYY-MM-DD with a real calendar date; else None.

    Models occasionally return ``"Q3 2024"``, ``"2024"``, ``"2024-13-01"``,
    or empty strings. We'd rather store NULL than something that looks
    structured but isn't.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not _ISO_DATE_RE.match(raw):
        return None
    try:
        date.fromisoformat(raw)
    except ValueError:
        return None
    return raw


class FilenameDateError(ValueError):
    """A `--from-filename` regex that can't extract dates (no `date` group / bad
    pattern). A *usage* error — distinct from a per-document non-match, which is
    just a count. Raised once, up front, before any document is examined."""


def compile_filename_date_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a `--from-filename` regex, requiring a named ``date`` group.

    Colocated with :func:`normalize_authored_date` so a later metadata sub-issue
    (#48 seeds) can reuse this named-capture mechanism for non-date facets.
    Raises :class:`FilenameDateError` on a malformed pattern or a pattern with no
    ``date`` group — both are operator mistakes worth refusing before a 174k-doc
    scan rather than silently matching nothing.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise FilenameDateError(f"Invalid regex {pattern!r}: {e}") from e
    if "date" not in compiled.groupindex:
        raise FilenameDateError(
            f"Regex {pattern!r} has no named group 'date'; "
            r"use e.g. '(?P<date>\d{4}-\d{2}-\d{2})'."
        )
    return compiled


def extract_filename_date(compiled: "re.Pattern[str]", name: str) -> str | None:
    """Return the raw substring captured by the ``date`` group, or None if the
    regex doesn't match ``name``. The raw value is *not* normalized here — the
    caller runs it through :func:`normalize_authored_date` so a match that
    captures a non-date is counted as invalid, never silently stored as NULL.
    """
    m = compiled.search(name)
    if m is None:
        return None
    return m.group("date")


@dataclass
class SummaryResult:
    title: str
    description: str
    text: str
    model: str
    authored_date: str | None  # ISO 8601, NULL when not stated or malformed


@lru_cache(maxsize=1)
def _tokenizer():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    tok = _tokenizer()
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text, len(ids)
    truncated = tok.decode(ids[:max_tokens])
    return truncated, len(ids)


def _truncation_note(used: int, total: int) -> str:
    return (
        f"\n\n_Note: this summary is based on the first {used:,} tokens "
        f"of a {total:,}-token document._"
    )


def summarize(
    document_text: str,
    *,
    provider: Provider,
    model: str,
    temperature: float,
    max_summarize_tokens: int,
    reasoning_effort: str | None = None,
) -> SummaryResult:
    """Summarize ``document_text``, truncating input if needed.

    Returns a ``SummaryResult`` whose ``text`` already includes the
    truncation note when applicable. The caller persists it verbatim.
    """
    if not document_text or not document_text.strip():
        raise ValueError("document_text is empty")

    input_text, total_tokens = _truncate_to_tokens(document_text, max_summarize_tokens)
    truncated = total_tokens > max_summarize_tokens

    summary: DocumentSummary = provider.summarize(
        input_text, model=model, temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    text = summary.text
    if truncated:
        text = text.rstrip() + _truncation_note(max_summarize_tokens, total_tokens)

    return SummaryResult(
        title=summary.title,
        description=summary.description,
        text=text,
        model=model,
        authored_date=normalize_authored_date(summary.authored_date),
    )
