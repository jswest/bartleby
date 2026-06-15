"""`bartleby finding` — out-of-band Markdown share of a single finding.

Two halves of one artifact format:

- ``export <finding-id>`` reads a finding from a corpus and writes a
  self-describing ``.md``: a YAML front-matter block (title, description, and
  baked-in provenance — the source corpus, the original finding id, and the
  export date) followed by the finding body. The body's corpus ``[^N]`` chunk
  citations are rewritten *inline* as inert ``[corpus: <file> · p.<N>]`` markers
  so the artifact stands alone on another machine where the raw chunk_ids are
  meaningless. Finding-to-finding citations are inlined the same way (carrying
  the cited finding's title). External ``[^url:…]`` / ``[^doc:…]`` markers are
  left verbatim — they're already self-describing.

- ``import <path>`` parses such an artifact and writes it into the active (or
  ``--project``) corpus through the same finding write path ``save_finding``
  uses (chunk + embed the body, insert via the typed chunk helpers). Provenance
  is prepended to the body as a human-readable header line, so it survives
  with the finding even though ``findings`` has no author/origin column (no
  schema change). The corpus citations stay inert ``[corpus: …]`` markers; they
  are deliberately NOT re-resolved to local chunk_ids (a raw chunk_id has no
  meaning across machines — the cross-machine natural-key resolver is a
  deferred follow-up). An imported finding renders like any other local finding
  in the read-only web viewer.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import apsw
import yaml
from rich.console import Console

from bartleby.db.connection import open_db
from bartleby.lib import console
from bartleby.project import get_active_project


_console = Console()

# Local chunk citation: ``[^N]`` (standard footnote with a chunk_id). Same shape
# the finding write path recognises (``_common._CITATION_MARKER``).
_CHUNK_MARKER = re.compile(r"\[\^(\d+)\]")
# An inert corpus citation an export emits / an import preserves verbatim. It is
# deliberately NOT a ``[^…]`` footnote (so the finding write path's malformed-
# and external-citation guards never trip on it) and carries no digits in the
# leading bracket (so it can't be mistaken for a ``[N]`` malformed marker).
_INERT_PREFIX = "[corpus: "


def _open_or_exit(corpus: str):
    """Open a corpus DB or exit(1) with a one-line error (no traceback)."""
    try:
        return open_db(corpus)
    except (RuntimeError, FileNotFoundError, apsw.Error) as e:
        console.error(str(e))
        sys.exit(1)


def _slug(title: str) -> str:
    """Filename slug from a title — mirrors the web viewer's download logic."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "finding"


def _inert_marker(citation: dict) -> str:
    """An inert, human-readable ``[corpus: …]`` marker for one resolved citation.

    Carries ``file · page`` for corpus chunks (the metadata a reader needs once
    the chunk_id is meaningless), or the finding title for finding-kind
    citations. Never a live link — see the module docstring.
    """
    if citation["source_kind"] == "finding":
        label = citation["source_name"] or f"finding {citation['chunk_id']}"
        return f"{_INERT_PREFIX}finding · {label}]"
    file_name = citation["file_name"] or citation["source_name"] or "unknown source"
    page = citation["page_number"]
    page_part = f" · p.{page}" if page is not None else ""
    return f"{_INERT_PREFIX}{file_name}{page_part}]"


def _rewrite_citations(body: str, citations: list[dict]) -> str:
    """Replace each ``[^N]`` marker with its inert ``[corpus: …]`` equivalent.

    ``citations`` is the resolved-citation list (chunk_id → file/page/title) the
    read path returns. A marker whose chunk_id isn't in the resolved set (a
    dangling citation — the source was removed) is left as a bare
    ``[corpus: source no longer available]`` so the provenance fact survives.
    """
    by_id = {c["chunk_id"]: c for c in citations}

    def _sub(m: re.Match) -> str:
        chunk_id = int(m.group(1))
        citation = by_id.get(chunk_id)
        if citation is None:
            return f"{_INERT_PREFIX}source no longer available]"
        return _inert_marker(citation)

    return _CHUNK_MARKER.sub(_sub, body)


def _read_finding_for_export(conn, finding_id: int) -> dict:
    """Read one finding (title/description/body + resolved citations) for export.

    Mirrors ``read_finding``'s reads but stays CLI-side and does no memory-wall
    check — export is a human operation over the local corpus, not an agent
    session.
    """
    from bartleby.skill_scripts._common import (
        finding_chunk_and_citation_ids,
        resolve_citations,
    )

    cur = conn.cursor()
    row = cur.execute(
        "SELECT title, description, body FROM findings WHERE finding_id = ?",
        (finding_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No finding with id {finding_id} in this corpus.")
    title, description, body = row
    _chunk_ids, citation_ids = finding_chunk_and_citation_ids(cur, finding_id)
    return {
        "title": title,
        "description": description,
        "body": body,
        "citations": resolve_citations(conn, citation_ids),
    }


def export(*, finding_id: int, project: str | None, out: str | None) -> None:
    """Emit a self-describing ``.md`` artifact for one finding.

    The artifact carries everything ``import`` needs to stand alone: YAML
    front-matter (title/description + provenance) and the body with corpus
    citations rewritten inline as inert ``[corpus: …]`` markers.
    """
    corpus = project or get_active_project()
    if not corpus:
        console.error("No active project. Specify one with --project.")
        sys.exit(1)

    conn = _open_or_exit(corpus)
    try:
        finding = _read_finding_for_export(conn, finding_id)
    except ValueError as e:
        console.error(str(e))
        sys.exit(1)
    finally:
        conn.close()

    rewritten = _rewrite_citations(finding["body"], finding["citations"])
    front_matter = {
        "title": finding["title"],
        "description": finding["description"],
        "provenance": {
            "source_corpus": corpus,
            "source_finding_id": finding_id,
            "exported_on": date.today().isoformat(),
        },
    }
    artifact = (
        "---\n"
        + yaml.safe_dump(front_matter, default_flow_style=False, sort_keys=False)
        + "---\n\n"
        + rewritten.rstrip("\n")
        + "\n"
    )

    out_path = Path(out) if out else Path(f"{_slug(finding['title'])}.md")
    out_path.write_text(artifact, encoding="utf-8")
    _console.print(
        f"[bold green]Exported finding #{finding_id}[/bold green] "
        f"from [cyan]{corpus}[/cyan] to [cyan]{out_path}[/cyan]"
    )


# A front-matter document is a leading ``---`` line, the YAML block, a closing
# ``---`` line, then the body. Captured non-greedily so a ``---`` inside the body
# (a markdown horizontal rule) doesn't terminate the block early.
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)


def parse_artifact(text: str) -> dict:
    """Parse an exported artifact into ``{title, description, provenance, body}``.

    Raises ``ValueError`` on a missing/blank front-matter block or a missing
    title — the artifact is the contract, and a malformed one is a user error,
    not a traceback.
    """
    m = _FRONT_MATTER.match(text)
    if not m:
        raise ValueError(
            "Not a Bartleby finding artifact: missing the leading YAML "
            "front-matter block (--- ... ---)."
        )
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed front-matter YAML: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError("Front-matter must be a YAML mapping.")

    title = (meta.get("title") or "").strip()
    description = (meta.get("description") or "").strip()
    body = m.group(2).strip()
    if not title:
        raise ValueError("Artifact front-matter is missing a non-empty title.")
    if not body:
        raise ValueError("Artifact has an empty body.")
    return {
        "title": title,
        "description": description,
        "provenance": meta.get("provenance") or {},
        "body": body,
    }


def _provenance_header(provenance: dict) -> str:
    """A human-readable provenance line to prepend to an imported body.

    ``findings`` has no author/origin column, so origin must live in the text.
    Falls back gracefully when a field is absent (a hand-written artifact)."""
    corpus = provenance.get("source_corpus")
    orig = provenance.get("source_finding_id")
    exported = provenance.get("exported_on")
    bits = []
    if corpus:
        bits.append(f"corpus `{corpus}`")
    if orig is not None:
        bits.append(f"orig. finding #{orig}")
    if exported:
        bits.append(f"exported {exported}")
    where = ", ".join(bits) if bits else "an external artifact"
    return f"_Imported from {where}._"


def import_(*, path: str, project: str | None) -> None:
    """Import a finding artifact into a corpus through the finding write path.

    Provenance is baked into the body as a header line (no schema change);
    corpus citations stay inert ``[corpus: …]`` markers (not re-resolved to
    local chunk_ids — see the module docstring).
    """
    from bartleby.session import ensure_active_session
    from bartleby.skill_scripts._common import (
        embed_body_chunks,
        write_finding_chunks,
    )

    artifact_path = Path(path)
    if not artifact_path.is_file():
        console.error(f"Artifact not found: {artifact_path}")
        sys.exit(1)
    try:
        parsed = parse_artifact(artifact_path.read_text(encoding="utf-8"))
    except ValueError as e:
        console.error(str(e))
        sys.exit(1)

    corpus = project or get_active_project()
    if not corpus:
        console.error("No active project. Specify one with --project.")
        sys.exit(1)

    # Bake provenance into the body so it travels with the finding (no
    # author/origin column exists). A blank line separates the header from the
    # original prose.
    body = f"{_provenance_header(parsed['provenance'])}\n\n{parsed['body']}"

    conn = _open_or_exit(corpus)

    # Embed before opening the write (same ordering rationale as save_finding).
    chunk_inputs = embed_body_chunks(body)
    try:
        session_id = ensure_active_session(corpus)
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                "INSERT INTO findings (session_id, title, description, body) "
                "VALUES (?, ?, ?, ?)",
                (session_id, parsed["title"], parsed["description"], body),
            )
            finding_id = conn.last_insert_rowid()
            # Corpus citations are inert markers, so there are no live chunk
            # citations to record — only the body chunks are written.
            write_finding_chunks(conn, finding_id, chunk_inputs)
        except BaseException:
            cur.execute("ROLLBACK")
            raise
        else:
            cur.execute("COMMIT")
    finally:
        conn.close()

    _console.print(
        f"[bold green]Imported finding #{finding_id}[/bold green] "
        f"into [cyan]{corpus}[/cyan] (provenance baked into the body)"
    )
