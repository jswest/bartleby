"""`bartleby scribe` — sequential ingest pipeline.

Hash → archive → convert → embed → (optionally) summarize → write. All chunk
writes go through ``bartleby.db.chunks`` typed helpers.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from bartleby.config import ensure_provider_env, load_config
from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_document_chunks,
    insert_summary_chunks,
)
from bartleby.db.connection import open_db
from bartleby.ingest.chunk import (
    SUPPORTED_EXTENSIONS,
    ChunkRow,
    chunk_markdown_string,
    convert_and_chunk,
)
from bartleby.ingest.embed import embed_texts
from bartleby.ingest.summarize import SummaryResult, count_tokens, summarize
from bartleby.lib import console
from bartleby.project import get_active_project, get_project_dir
from bartleby.providers import Provider, get_provider


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _archive(src: Path, archive_root: Path, file_hash: str) -> Path:
    ext = src.suffix.lower()
    dest_dir = archive_root / file_hash
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{file_hash}{ext}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise ValueError(f"Path not found: {path}")


def _build_chunk_inputs(
    rows: list[ChunkRow],
    embeddings: list[list[float]],
    start_index: int = 0,
) -> list[ChunkInput]:
    return [
        ChunkInput(
            text=row.text,
            embedding=emb,
            chunk_index=start_index + i,
            section_heading=row.section_heading,
            content_type=row.content_type,
        )
        for i, (row, emb) in enumerate(zip(rows, embeddings))
    ]


def _document_already_ingested(conn, file_hash: str) -> int | None:
    row = conn.cursor().execute(
        "SELECT document_id FROM documents WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    return row[0] if row else None


def _write_summary(conn, document_id: int, summary: SummaryResult) -> int:
    cur = conn.cursor()
    prior = cur.execute(
        "SELECT summary_id FROM summaries WHERE document_id = ?", (document_id,)
    ).fetchone()
    if prior:
        delete_chunks_for(conn, "summary", prior[0])
        cur.execute("DELETE FROM summaries WHERE summary_id = ?", (prior[0],))

    cur.execute(
        "INSERT INTO summaries "
        "(document_id, title, description, text, model) "
        "VALUES (?, ?, ?, ?, ?)",
        (document_id, summary.title, summary.description,
         summary.text, summary.model),
    )
    summary_id = conn.last_insert_rowid()

    rows = chunk_markdown_string(summary.text)
    if rows:
        embeddings = embed_texts([r.text for r in rows])
        insert_summary_chunks(conn, summary_id, _build_chunk_inputs(rows, embeddings))
    return summary_id


def _maybe_summarize(
    conn,
    document_id: int,
    full_text: str,
    *,
    provider: Provider | None,
    model: str | None,
    temperature: float,
    max_summarize_tokens: int,
) -> int | None:
    if provider is None or not model:
        return None
    result = summarize(
        full_text,
        provider=provider,
        model=model,
        temperature=temperature,
        max_summarize_tokens=max_summarize_tokens,
    )
    return _write_summary(conn, document_id, result)


def _process_one(
    conn,
    path: Path,
    archive_root: Path,
    *,
    provider: Provider | None,
    model: str | None,
    temperature: float,
    max_summarize_tokens: int,
) -> None:
    file_hash = _hash_file(path)
    if _document_already_ingested(conn, file_hash) is not None:
        console.info(f"Skipping {path.name} (already ingested)")
        return

    archived = _archive(path, archive_root, file_hash)
    result = convert_and_chunk(archived)

    if not result.full_text.strip() or not result.chunks:
        console.warn(f"{path.name}: no extractable text; skipping")
        return

    token_count = count_tokens(result.full_text)
    conn.cursor().execute(
        "INSERT INTO documents "
        "(file_hash, file_name, file_path, page_count, token_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_hash, path.name, str(archived), result.page_count, token_count),
    )
    document_id = conn.last_insert_rowid()

    embeddings = embed_texts([row.text for row in result.chunks])
    insert_document_chunks(
        conn, document_id, _build_chunk_inputs(result.chunks, embeddings)
    )

    _maybe_summarize(
        conn, document_id, result.full_text,
        provider=provider, model=model,
        temperature=temperature, max_summarize_tokens=max_summarize_tokens,
    )


def _resolve_provider(
    config: dict,
    *,
    provider_override: str | None,
    model_override: str | None,
) -> tuple[Provider | None, str | None]:
    if config.get("summary_depth", "none") != "one-shot":
        return None, None

    name = provider_override or config.get("provider")
    model = model_override or config.get("model")
    if not name or not model:
        console.warn(
            "summary_depth=one-shot but no provider/model configured; "
            "skipping summaries."
        )
        return None, None

    ensure_provider_env(name, config)
    return get_provider(name, ollama_base_url=config.get("ollama_base_url")), model


def main(
    *,
    project: str | None,
    files: str | Path,
    model: str | None = None,
    provider: str | None = None,
    verbose: bool = False,
) -> None:
    from bartleby.lib.quiet import setup_quiet_third_party
    setup_quiet_third_party(verbose=verbose)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    project_name = project or get_active_project()
    if not project_name:
        raise RuntimeError(
            "No active project. Run `bartleby project create <name>` first."
        )

    config = load_config()
    provider_obj, model = _resolve_provider(
        config, provider_override=provider, model_override=model,
    )

    sources = _collect_files(Path(files))
    if not sources:
        console.warn("No supported files found.")
        return

    archive_root = get_project_dir(project_name) / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    console.big(f"Ingesting {len(sources)} file(s) into project '{project_name}'")

    conn = open_db(project_name)
    try:
        temperature = float(config.get("temperature", 0))
        max_summarize_tokens = int(config.get("max_summarize_tokens", 50_000))

        # Render progress on stderr so we can safely capture stdout from
        # Docling/RapidOCR around convert() without breaking the bar.
        with Progress(
            TextColumn("[bold]{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=Console(file=sys.stderr),
        ) as bar:
            task = bar.add_task("files", total=len(sources), label="Ingesting")
            for src in sources:
                bar.update(task, label=f"Ingesting {src.name}")
                try:
                    _process_one(
                        conn, src, archive_root,
                        provider=provider_obj, model=model,
                        temperature=temperature,
                        max_summarize_tokens=max_summarize_tokens,
                    )
                except Exception as e:
                    console.error(f"Failed: {src.name}: {e}")
                    if verbose:
                        logger.exception(e)
                finally:
                    bar.advance(task)
    finally:
        conn.close()

    console.complete("Done.")
