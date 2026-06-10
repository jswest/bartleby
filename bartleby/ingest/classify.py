"""Source classification for the scribe ingest run.

Pulled out of ``commands/scribe.py`` (#306): walking the supplied paths into
ingestible ``(file, ext)`` pairs (:func:`_collect_files`), hashing each, and
bucketing it against DB state (:func:`_classify`) — skip already-done,
resume parsed-but-uncaptioned, queue never-parsed, divert in-run duplicates.
All of this runs on the main process; only the queued :class:`ParseRequest`
bucket crosses to the parse workers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bartleby.ingest import parsers
from bartleby.ingest.chunk import IMAGE_EXTENSIONS, resolve_extension
from bartleby.ingest.writer import Writer
from bartleby.lib import console


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _collect_files(
    paths: list[Path], only: set[str] | None = None,
) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Resolve ``paths`` to ingestible ``(file, extension)`` pairs.

    Each path is a single file or a directory walked recursively; the results
    are concatenated and de-duplicated by real path, so a file reachable from
    two supplied roots (or named twice) is collected once (issue #89).

    The resolved extension is what the file should be *treated as* — the
    filename extension when it is supported, otherwise a content-sniffed one
    (see :func:`resolve_extension`). When ``only`` is given (a set of supported,
    leading-dot extensions), files whose resolved type is not in it are dropped
    from collection — silently, since the exclusion is intentional and not the
    same as a file we couldn't identify.

    Returns ``(sources, unidentified)`` where ``unidentified`` lists directory
    entries that could not be resolved to a supported type at all; the caller
    surfaces them so they are not dropped silently.
    """
    sources: list[tuple[Path, str]] = []
    unidentified: list[Path] = []
    seen: set[Path] = set()

    def _first_seen(p: Path) -> bool:
        key = p.resolve()
        if key in seen:
            return False
        seen.add(key)
        return True

    for path in paths:
        if path.is_file():
            if not _first_seen(path):
                continue
            ext = resolve_extension(path)
            if ext is None:
                raise ValueError(f"Unsupported file type: {path.name}")
            if only is None or ext in only:
                sources.append((path, ext))
        elif path.is_dir():
            for p in sorted(path.rglob("*")):
                if not p.is_file() or not _first_seen(p):
                    continue
                ext = resolve_extension(p)
                if ext is None:
                    unidentified.append(p)
                elif only is None or ext in only:
                    sources.append((p, ext))
        else:
            raise ValueError(f"Path not found: {path}")

    return sources, unidentified

def _is_complete(writer: Writer, document_id: int) -> bool:
    """Per-unit drain completeness: parsed ∧ every image captioned.

    Summaries are settled by their own pass (:func:`_summarize_all`), not
    here — a document missing only its summary is "complete" for the
    parse/caption drain and is picked up later from the DB by that pass.
    """
    return not writer.uncaptioned_images(document_id)

@dataclass
class _ResumeItem:
    """A document parsed by an earlier run that still has missing units."""
    document_id: int
    file_name: str
    file_hash: str

def _classify(
    writer: Writer,
    sources: list[tuple[Path, str]],
    *,
    vision_enabled: bool,
) -> tuple[list[parsers.ParseRequest], list[_ResumeItem], list[str], list[str]]:
    """Bucket each source by what the parse/caption drain still needs, all from
    DB state: already parsed+captioned (skip), parsed-but-uncaptioned (resume —
    no parse), or never parsed (hand to the pool). Summaries are not considered
    here — they're settled by their own pass over whatever the DB still lacks.
    Hashing + the resume lookups run here on the main process; only the parse
    bucket crosses to workers.

    A fourth bucket, ``duplicates``, catches two byte-identical files *within one
    run*: ``documents.file_hash`` is UNIQUE, so only the first can persist. The
    DB lookup can't see the in-run twin (neither is committed yet), so we track
    hashes queued this run and divert the rest here rather than let the second
    crash ``persist_parse`` (#225)."""
    to_parse: list[parsers.ParseRequest] = []
    to_resume: list[_ResumeItem] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    queued_hashes: set[str] = set()
    for path, ext in sources:
        file_hash = _hash_file(path)
        document_id = writer.document_id_for(file_hash)
        if document_id is not None:
            if _is_complete(writer, document_id):
                skipped.append(path.name)
            else:
                to_resume.append(_ResumeItem(document_id, path.name, file_hash))
            continue
        if file_hash in queued_hashes:
            duplicates.append(path.name)
            continue
        if ext in IMAGE_EXTENSIONS and not vision_enabled:
            console.warn(
                f"{path.name}: skipping image (no vision provider configured)."
            )
            continue
        queued_hashes.add(file_hash)
        to_parse.append(
            parsers.ParseRequest(
                path=path, ext=ext, file_hash=file_hash, file_name=path.name
            )
        )
    return to_parse, to_resume, skipped, duplicates
