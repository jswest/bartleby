from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bartleby.lib.config import load_config, setup_provider_env
from bartleby.lib.console import send
from bartleby.lib.consts import DEFAULT_MAX_WORKERS, DEFAULT_PDF_PAGES_TO_SUMMARIZE, DOCLING_MAX_TOKENS, EMBEDDING_MODEL
from bartleby.lib.utils import build_model_id, has_vision
from bartleby.read.converters import convert_html_to_pdf
from bartleby.read.processor import process_pdf, process_pdf_docling
from bartleby.read.sqlite import get_connection


def _process_pdf_worker(args):
    """
    Worker function for multiprocessing - creates its own connection, embedding model, and model_id.
    Must be at module level to be picklable.

    Args:
        args: Tuple of (pdf_file, db_path, archive_path, model_id, llm_has_vision,
                       pdf_pages_to_summarize, config, verbose)
    """
    pdf_file, db_path, archive_path, model_id, llm_has_vision, pdf_pages_to_summarize, config, verbose = args

    # Configure logging in worker process
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    # Each process gets its own connection and embedding model for isolation
    connection = get_connection(db_path)
    # Create embedding model in the worker process
    process_embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Set API keys from config in worker process
    setup_provider_env(config)

    try:
        process_pdf(
            connection,
            pdf_file,
            db_path,
            archive_path,
            process_embedding_model,
            model_id,
            llm_has_vision,
            pdf_pages_to_summarize
        )
    finally:
        connection.close()


def _process_docling(pdf_files, db_path, archive_path, model_id, llm_has_vision, pdf_pages_to_summarize):
    """Process documents sequentially using Docling's layout-aware pipeline."""
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    chunker = HybridChunker(tokenizer=EMBEDDING_MODEL, max_tokens=DOCLING_MAX_TOKENS)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    connection = get_connection(db_path)

    try:
        for pdf_file in tqdm(pdf_files, desc="Processing documents (Docling)", unit="doc"):
            try:
                process_pdf_docling(
                    connection,
                    pdf_file,
                    db_path,
                    archive_path,
                    embedding_model,
                    converter,
                    chunker,
                    model_id,
                    llm_has_vision,
                    pdf_pages_to_summarize,
                )
                logger.debug(f"Successfully processed: {pdf_file}")
            except Exception as e:
                send(f"Failed to process {pdf_file.name}: {e}", "ERROR")
    finally:
        connection.close()


def main(db_path, pdf_path, max_workers: int = None, model: str = None, provider: str = None, verbose: bool = False, use_docling: bool = False):
    # Configure logging level
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    # Load config from ~/.bartleby/config.yaml
    config = load_config()

    # Apply config defaults (CLI args take precedence)
    if max_workers is None:
        max_workers = config.get("max_workers", DEFAULT_MAX_WORKERS)
    if model is None:
        model = config.get("model")
    if provider is None:
        provider = config.get("provider")

    # Get summarization config
    pdf_pages_to_summarize = config.get("pdf_pages_to_summarize", DEFAULT_PDF_PAGES_TO_SUMMARIZE)

    # Set API keys from config
    setup_provider_env(config)

    db_path = Path(db_path)
    pdf_path = Path(pdf_path)
    archive_path = db_path.parent / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)

    send(f"Embedding model: {EMBEDDING_MODEL}", "BIG")

    # Build model_id for litellm
    model_id = ""
    llm_has_vision = False
    if model and provider:
        # Build config dict for build_model_id
        model_config = {
            "provider": provider,
            "model": model,
            "ollama_base_url": config.get("ollama_base_url", "http://localhost:11434"),
        }
        model_id = build_model_id(model_config) or ""
        llm_has_vision = has_vision(model_config)

        if model_id:
            send(f"Loading LLM: {model_id}", "BIG")
        else:
            send(f"Unknown provider: {provider}", "WARN")

    # Collect PDF and HTML files to process
    pdf_files = []
    html_files = []

    if pdf_path.is_file():
        if pdf_path.suffix.lower() == ".pdf":
            pdf_files = [pdf_path]
        elif pdf_path.suffix.lower() in [".html", ".htm"]:
            html_files = [pdf_path]
        else:
            raise ValueError(f"Unsupported file type: {pdf_path.suffix}")
    elif pdf_path.is_dir():
        pdf_files = list(pdf_path.rglob("*.pdf"))
        html_files = list(pdf_path.rglob("*.html")) + list(pdf_path.rglob("*.htm"))
    else:
        raise ValueError(f"Invalid pdf_path: {pdf_path}")

    # Convert HTML files to PDF
    temp_dir = None
    if html_files:
        send(f"Converting {len(html_files)} HTML file(s) to PDF", "BIG")
        temp_dir = Path(tempfile.mkdtemp(prefix="bartleby_html_"))

        for html_file in tqdm(html_files, desc="Converting HTML to PDF", unit="file"):
            try:
                pdf_file = convert_html_to_pdf(html_file, temp_dir)
                pdf_files.append(pdf_file)
            except Exception as e:
                send(f"Failed to convert {html_file.name}: {e}", "ERROR")

    if use_docling:
        send(f"Processing {len(pdf_files)} document(s) with Docling (sequential)", "BIG")
        _process_docling(pdf_files, db_path, archive_path, model_id, llm_has_vision, pdf_pages_to_summarize)
    else:
        send(f"Processing {len(pdf_files)} document(s) with {max_workers} workers", "BIG")

        # Use ProcessPoolExecutor instead of ThreadPoolExecutor for SentenceTransformer thread-safety
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare arguments for each worker (must be picklable - pass model_id string instead of LLM objects)
            worker_args = [
                (pdf_file, db_path, archive_path, model_id, llm_has_vision,
                 pdf_pages_to_summarize, config, verbose)
                for pdf_file in pdf_files
            ]

            futures = {
                executor.submit(_process_pdf_worker, args): args[0]
                for args in worker_args
            }

            with tqdm(total=len(pdf_files), desc="Processing documents", unit="doc") as pbar:
                for future in as_completed(futures):
                    pdf_file = futures[future]
                    try:
                        future.result()
                        logger.debug(f"Successfully processed: {pdf_file}")
                    except Exception as e:
                        send(f"Failed to process {pdf_file.name}: {e}", "ERROR")
                    finally:
                        pbar.update(1)

    # Clean up temporary directory if HTML files were converted
    if temp_dir and temp_dir.exists():
        logger.debug(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    send("Processing complete!", "COMPLETE")
