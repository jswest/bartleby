"""Interactive configuration wizard for Bartleby."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from bartleby.config import CONFIG_PATH, load_config, save_config
from bartleby.lib.consts import (
    DEFAULT_HTML_CONVERTER,
    DEFAULT_OCR_MIN_CONFIDENCE,
    DEFAULT_PDF_CONVERTER,
    DEFAULT_SPARSE_TEXT_THRESHOLD,
    DEFAULT_VISION_MAX_DIMENSION,
    DEFAULT_VISION_MIN_DIMENSION,
)
from bartleby.providers import ALLOWED_PROVIDERS

ALLOWED_SUMMARY_DEPTHS = ["none", "one-shot"]
ALLOWED_PDF_CONVERTERS = ["pdfplumber", "docling"]
ALLOWED_HTML_CONVERTERS = ["docling", "sec2md"]

PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "ollama": "qwen3-vl:30b",
    "wsjpt": "fast",
}

VISION_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "ollama": "qwen3-vl:30b",
    "wsjpt": "fast",
}

DEFAULT_SUMMARY_DEPTH = "one-shot"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_SUMMARIZE_TOKENS = 50_000
DEFAULT_MAX_READ_TOKENS = 50_000
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

console = Console()


def _help(text: str) -> None:
    """Print a concise, dim help blurb above the upcoming prompt.

    The wizard's prompt labels stay short (Rich renders them on one line); the
    orienting detail lives here, indented and dimmed so it reads as secondary.
    Multi-line strings are printed line-for-line.
    """
    for line in text.split("\n"):
        console.print(f"  {line}", style="dim")


def _prompt_provider(existing: dict, *, help_text: str,
                     label: str = "LLM provider",
                     key: str = "provider") -> str:
    default_provider = existing.get(key, "anthropic")
    if default_provider not in ALLOWED_PROVIDERS:
        default_provider = "anthropic"
    choices = " / ".join(ALLOWED_PROVIDERS)
    _help(help_text)
    while True:
        provider = Prompt.ask(
            f"{label} ({choices})", default=default_provider
        ).lower()
        if provider in ALLOWED_PROVIDERS:
            return provider
        console.print(f"[red]Invalid provider. Choose from: {choices}[/red]")


def _prompt_model(provider: str, existing: dict,
                  *, help_text: str, key: str = "model",
                  defaults: dict[str, str] = PROVIDER_DEFAULT_MODEL) -> str:
    default_model = existing.get(key) or defaults[provider]
    _help(help_text)
    return Prompt.ask("Model name", default=default_model)


def _prompt_api_key(provider: str, existing: dict, *, help_text: str | None = None) -> str | None:
    field = f"{provider}_api_key"
    existing_key = existing.get(field, "")
    _help(help_text or (
        "Saved to the config file. Leave blank to fall back to the "
        f"{provider.upper()}_API_KEY environment variable."
    ))
    if existing_key:
        if Confirm.ask("API key already configured. Update it?", default=False):
            entered = Prompt.ask("API key", password=True)
            return entered or existing_key
        return existing_key
    entered = Prompt.ask(
        "API key (optional)", password=True, default=""
    )
    return entered or None


def _prompt_pdf_converter(existing: dict, *, help_text: str) -> str:
    default = existing.get("pdf_converter", DEFAULT_PDF_CONVERTER)
    if default not in ALLOWED_PDF_CONVERTERS:
        default = DEFAULT_PDF_CONVERTER
    choices = " / ".join(ALLOWED_PDF_CONVERTERS)
    _help(help_text)
    while True:
        b = Prompt.ask(
            f"PDF converter ({choices})", default=default
        ).lower()
        if b in ALLOWED_PDF_CONVERTERS:
            return b
        console.print(f"[red]Invalid converter. Choose from: {choices}[/red]")


def _prompt_html_converter(existing: dict, *, help_text: str) -> str:
    default = existing.get("html_converter", DEFAULT_HTML_CONVERTER)
    if default not in ALLOWED_HTML_CONVERTERS:
        default = DEFAULT_HTML_CONVERTER
    choices = " / ".join(ALLOWED_HTML_CONVERTERS)
    _help(help_text)
    while True:
        b = Prompt.ask(
            f"HTML converter ({choices})", default=default
        ).lower()
        if b in ALLOWED_HTML_CONVERTERS:
            return b
        console.print(f"[red]Invalid converter. Choose from: {choices}[/red]")


def _prompt_summary_depth(existing: dict, *, help_text: str) -> str:
    default = existing.get("summary_depth", DEFAULT_SUMMARY_DEPTH)
    choices = " / ".join(ALLOWED_SUMMARY_DEPTHS)
    _help(help_text)
    while True:
        depth = Prompt.ask(
            f"Summary depth ({choices})", default=default
        ).lower()
        if depth in ALLOWED_SUMMARY_DEPTHS:
            return depth
        console.print(f"[red]Invalid summary depth. Choose from: {choices}[/red]")


def _prompt_temperature(existing: dict, *, help_text: str) -> float:
    default = float(existing.get("temperature", DEFAULT_TEMPERATURE))
    _help(help_text)
    while True:
        t = FloatPrompt.ask("Temperature", default=default)
        if 0 <= t <= 1:
            return t
        console.print("[red]Temperature must be between 0 and 1[/red]")


def _prompt_ollama_url(existing: dict) -> str:
    _help("Where your local Ollama server listens.")
    return Prompt.ask(
        "Ollama server URL",
        default=existing.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
    )


def _prompt_positive_int(prompt: str, default: int, *, help_text: str) -> int:
    _help(help_text)
    while True:
        n = IntPrompt.ask(prompt, default=default)
        if n > 0:
            return n
        console.print("[red]Must be a positive integer[/red]")


def _prompt_max_workers(existing: dict) -> int | None:
    """Returns the worker count, or None for auto (omit the key from config)."""
    _help(
        "How many documents scribe parses in parallel. 0 = auto: the min of your "
        "CPU cores and what free RAM allows.\nRaise it for a faster bulk ingest "
        "on a big machine; lower it if memory is tight."
    )
    current = existing.get("max_workers")
    while True:
        n = IntPrompt.ask("Parse workers (0 = auto)", default=int(current) if current else 0)
        if n >= 0:
            return n or None
        console.print("[red]Must be 0 (auto) or a positive integer[/red]")


def main():
    console.print(Panel.fit(
        "[bold cyan]Bartleby Configuration[/bold cyan]\n"
        "Let's set up your preferences and API keys.",
        border_style="cyan",
    ))

    existing = load_config()
    # Rebuild the config from prompts; only preserve project-pointer state.
    config: dict = {}
    if existing.get("active_project"):
        config["active_project"] = existing["active_project"]

    console.print("\n[bold]LLM Configuration[/bold] (for ingest-time summarization)")
    _help(
        "An LLM summarizes each document at ingest. Summaries are searchable "
        "and shown when browsing.\nSkip this to ingest and index raw text only."
    )
    use_llm = Confirm.ask(
        "Do you want to configure an LLM provider?",
        default=bool(existing.get("provider")),
    )

    if use_llm:
        provider = _prompt_provider(
            existing,
            help_text="The service that runs your summarization model.",
        )
        config["provider"] = provider
        config["model"] = _prompt_model(
            provider, existing,
            help_text="The summarization model. The default is a fast, "
            "low-cost choice for this provider.",
        )

        if provider in ("anthropic", "openai", "wsjpt"):
            key = _prompt_api_key(provider, existing)
            field = f"{provider}_api_key"
            if key:
                config[field] = key
            else:
                config.pop(field, None)
        elif provider == "ollama":
            config["ollama_base_url"] = _prompt_ollama_url(existing)

        console.print("\n[bold]Summarization[/bold]")
        depth = _prompt_summary_depth(
            existing,
            help_text="none = skip summaries; one-shot = one summary per "
            "document.\nSummaries are indexed for search/scan and shown in listings.",
        )
        config["summary_depth"] = depth
        if depth == "one-shot":
            config["temperature"] = _prompt_temperature(
                existing,
                help_text="0 = deterministic, repeatable summaries; higher = "
                "more varied wording.\nLeave at 0 unless summaries feel too rigid.",
            )
            config["max_summarize_tokens"] = _prompt_positive_int(
                "Max summarization input tokens",
                int(existing.get("max_summarize_tokens", DEFAULT_MAX_SUMMARIZE_TOKENS)),
                help_text="Caps how much document text is sent to the summarizer; "
                "longer documents are truncated.\nHigher = more context, higher cost.",
            )
        else:
            config.pop("temperature", None)
            config.pop("max_summarize_tokens", None)
    else:
        # No LLM → no summarization.
        for k in ("provider", "model", "summary_depth", "temperature",
                 "max_summarize_tokens", "ollama_base_url",
                 "anthropic_api_key", "openai_api_key", "wsjpt_api_key"):
            config.pop(k, None)
        config["summary_depth"] = "none"

    console.print("\n[bold]Converters[/bold]")
    config["pdf_converter"] = _prompt_pdf_converter(
        existing,
        help_text="pdfplumber = fast text extraction; docling = slower but "
        "layout-aware (tables, columns, reading order).",
    )
    config["html_converter"] = _prompt_html_converter(
        existing,
        help_text="docling handles general HTML/Markdown; sec2md is specialized "
        "for iXBRL EDGAR filings (preserves SEC tables/headings, others stay on "
        "docling).",
    )
    config["sparse_text_threshold"] = _prompt_positive_int(
        "Sparse-text threshold",
        int(existing.get("sparse_text_threshold", DEFAULT_SPARSE_TEXT_THRESHOLD)),
        help_text="Pages with fewer than this many extracted characters are "
        "treated as scanned and routed to OCR/VLM.\nLower = more pages OCR'd "
        "(slower, catches more).",
    )
    max_workers = _prompt_max_workers(existing)
    if max_workers is not None:
        config["max_workers"] = max_workers

    console.print("\n[bold]Image analysis[/bold] (VLM captions/OCR for embedded + standalone images)")
    _help(
        "A vision model captions and OCRs images (embedded figures and "
        "standalone image files).\nSkip to leave images unanalyzed."
    )
    use_vision = Confirm.ask(
        "Configure a vision provider for image analysis?",
        default=bool(existing.get("vision_provider")),
    )
    if use_vision:
        vprovider = _prompt_provider(
            existing, label="Vision provider", key="vision_provider",
            help_text="The service that runs your vision (image) model.",
        )
        config["vision_provider"] = vprovider
        config["vision_model"] = _prompt_model(
            vprovider, existing,
            key="vision_model", defaults=VISION_PROVIDER_DEFAULT_MODEL,
            help_text="The vision model used to caption/OCR images. The default "
            "is a fast, low-cost choice for this provider.",
        )
        # If the vision provider is the same as the LLM provider, reuse the
        # api key already prompted for above. Otherwise prompt fresh.
        if vprovider in ("anthropic", "openai", "wsjpt"):
            if vprovider != config.get("provider"):
                key = _prompt_api_key(vprovider, existing)
                field = f"{vprovider}_api_key"
                if key:
                    config[field] = key
        elif vprovider == "ollama":
            # Reuse ollama_base_url if already set; otherwise prompt.
            if "ollama_base_url" not in config:
                config["ollama_base_url"] = _prompt_ollama_url(existing)
        config["vision_max_dimension"] = _prompt_positive_int(
            "Max image dimension (px)",
            int(existing.get("vision_max_dimension", DEFAULT_VISION_MAX_DIMENSION)),
            help_text="Images are downscaled so their long edge is at most this "
            "many pixels before the VLM sees them.\nLower = faster/cheaper, less detail.",
        )
        config["vision_min_dimension"] = _prompt_positive_int(
            "Min image dimension (px)",
            int(existing.get("vision_min_dimension", DEFAULT_VISION_MIN_DIMENSION)),
            help_text="Images with an edge smaller than this are skipped — "
            "avoids wasting VLM calls (and crashes) on thin slivers.",
        )
        config["ocr_min_confidence"] = _prompt_positive_int(
            "Tesseract min confidence",
            int(existing.get("ocr_min_confidence", DEFAULT_OCR_MIN_CONFIDENCE)),
            help_text="Tesseract average confidence (0-100); pages scoring below "
            "this fall back to the VLM.\nHigher = trust OCR less, use the VLM more.",
        )
    else:
        for k in ("vision_provider", "vision_model",
                 "vision_max_dimension", "vision_min_dimension",
                 "ocr_min_confidence"):
            config.pop(k, None)

    console.print("\n[bold]Document reading[/bold]")
    config["max_read_tokens"] = _prompt_positive_int(
        "Max read tokens",
        int(existing.get("max_read_tokens", DEFAULT_MAX_READ_TOKENS)),
        help_text="An agent's `read_document` must pass --force to pull a "
        "document larger than this many tokens.\nGuards against blowing the "
        "context window on a huge file.",
    )

    save_config(config)

    console.print("\n[bold green]Configuration saved.[/bold green]")
    console.print(f"Config location: [cyan]{CONFIG_PATH}[/cyan]")

    if not config.get("active_project"):
        console.print(
            "\n[yellow]Tip:[/yellow] Create a project with "
            "[bold]bartleby project create <name>[/bold]"
        )
