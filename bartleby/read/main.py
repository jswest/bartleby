from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bartleby.lib.console import send
from bartleby.lib.consts import DEFAULT_MAX_WORKERS, EMBEDDING_MODEL
from bartleby.lib.utils import load_config
from bartleby.read.processor import process_pdf
from bartleby.read.sqlite import get_connection


def _process_pdf_worker(args):
    """
    Worker function for multiprocessing - creates its own connection, embedding model, and LLM.
    Must be at module level to be picklable.

    Args:
        args: Tuple of (pdf_file, db_path, archive_path, provider, model, llm_has_vision,
                       pdf_pages_to_summarize, config, verbose)
    """
    pdf_file, db_path, archive_path, provider, model, llm_has_vision, pdf_pages_to_summarize, config, verbose = args

    # Configure logging in worker process
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Each process gets its own connection and embedding model for isolation
    connection = get_connection(db_path)
    # Create embedding model in the worker process
    process_embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Create LLM in worker process (can't pickle LLM objects due to thread locks)
    llm = None
    if model and provider:
        # Set API keys from config if present
        if provider:
            api_key_field = f"{provider}_api_key"
            config_api_key = config.get(api_key_field)
            env_var_name = f"{provider.upper()}_API_KEY"

            if config_api_key and not os.environ.get(env_var_name):
                os.environ[env_var_name] = config_api_key

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            base_url = config.get("ollama_base_url", "http://localhost:11434")
            llm = ChatOllama(model=model, base_url=base_url)

    try:
        process_pdf(
            connection,
            pdf_file,
            db_path,
            archive_path,
            process_embedding_model,
            llm,
            llm_has_vision,
            pdf_pages_to_summarize
        )
    finally:
        connection.close()


def main(db_path, pdf_path, max_workers: int = None, model: str = None, provider: str = None, verbose: bool = False):
    # Configure logging level
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

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
    from bartleby.lib.consts import DEFAULT_PDF_PAGES_TO_SUMMARIZE
    pdf_pages_to_summarize = config.get("pdf_pages_to_summarize", DEFAULT_PDF_PAGES_TO_SUMMARIZE)

    # Set API keys from config if present and not already in environment
    if provider:
        api_key_field = f"{provider}_api_key"
        config_api_key = config.get(api_key_field)
        env_var_name = f"{provider.upper()}_API_KEY"

        if config_api_key and not os.environ.get(env_var_name):
            os.environ[env_var_name] = config_api_key
            logger.debug(f"Using {provider} API key from config")

    db_path = Path(db_path)
    pdf_path = Path(pdf_path)
    archive_path = db_path.parent / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)

    send(f"Embedding model: {EMBEDDING_MODEL}", "BIG")

    llm = None
    llm_has_vision = False
    if model and provider:
        send(f"Loading LLM: {provider}/{model}", "BIG")
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model)
            llm_has_vision = "claude-3" in model or "claude-4" in model
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model)
            llm_has_vision = "gpt-4" in model and "vision" in model
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            base_url = config.get("ollama_base_url", "http://localhost:11434")
            llm = ChatOllama(model=model, base_url=base_url)
            llm_has_vision = False  # Ollama typically doesn't support vision
        else:
            send(f"Unknown provider: {provider}", "WARN")

    # Collect PDF files to process
    pdf_files = []
    if pdf_path.is_file():
        pdf_files = [pdf_path]
    elif pdf_path.is_dir():
        pdf_files = list(pdf_path.rglob("*.pdf"))
    else:
        raise ValueError(f"Invalid pdf_path: {pdf_path}")

    send(f"Processing {len(pdf_files)} document(s) with {max_workers} workers", "BIG")

    # Use ProcessPoolExecutor instead of ThreadPoolExecutor for SentenceTransformer thread-safety
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each worker (must be picklable - can't pass LLM objects)
        worker_args = [
            (pdf_file, db_path, archive_path, provider, model, llm_has_vision,
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

    send("Processing complete!", "COMPLETE")