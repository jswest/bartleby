"""Constants used across the v1 ingest pipeline.

Other knobs (chunker limits, search defaults) live in the module that owns
them — keep this file focused on shared values.
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Hugging Face repos Docling lazily fetches for its default PDF/HTML pipeline:
# the layout model (analysed on every page) and the TableFormer model (pages
# with tables). Both download on demand, the layout model not until the first
# docling conversion. Docling's OCR is configured "auto", which resolves to a
# non-HF engine (Apple Vision / bundled RapidOCR), so it adds nothing here.
# These gate offline mode (see lib/quiet.py): until both are cached, ingest
# stays online so they can download. Version-coupled to Docling — if a docling
# bump changes its model repos, update this list.
DOCLING_HF_REPOS = (
    "docling-project/docling-layout-heron",
    "docling-project/docling-models",
)

# Ingest conversion / vision knobs shared by the `config` wizard (which writes
# them into config) and `scribe` (which falls back to the DEFAULT_* values when a
# knob is absent, and whose argparse `choices=` are built from the ALLOWED_*
# lists). Sourcing both surfaces from here keeps them in lockstep — the values
# scribe accepts can never drift from the ones the wizard validates. This module
# imports nothing, so `cli.py` can read these at parser-build time without paying
# the provider package's pydantic import cost on every invocation.
ALLOWED_PROVIDERS = ("anthropic", "openai", "ollama", "wsjpt")
ALLOWED_PDF_CONVERTERS = ["pdfplumber", "docling"]
ALLOWED_HTML_CONVERTERS = ["docling", "sec2md"]
DEFAULT_PDF_CONVERTER = "pdfplumber"
DEFAULT_HTML_CONVERTER = "docling"
DEFAULT_SPARSE_TEXT_THRESHOLD = 100
DEFAULT_OCR_MIN_CONFIDENCE = 30
DEFAULT_VISION_MAX_DIMENSION = 768
# VLM image processors (e.g. qwen3-vl) tile images into fixed-size patches and
# crash when an edge is smaller than the patch factor (32 for qwen3-vl) — a hard
# floor that scales up for models with larger patches. Images below the default
# on either edge — thin rules, banners, sliver crops — are skipped before the VLM
# call. The default sits above that floor at 64 to also drop tiny crops the VLM
# can't read but will "describe" with confident nonsense. Raise it further for
# noisier corpora.
DEFAULT_VISION_MIN_DIMENSION = 64

# Parse-pool sizing (#165). When `max_workers` is unset, scribe auto-picks
# min(cpu_count - RESERVED_CORES, free_ram_gb // PER_WORKER_GB), floored at 1 —
# so a box that's CPU-rich but RAM-poor doesn't launch more parse workers than
# memory can hold and OOM, and the auto-pick always leaves a couple of cores for
# the OS and the rest of the machine instead of pinning every core for hours.
# Each worker loads the embedding model and, for docling ingests, the layout/table
# models; PER_WORKER_GB is a deliberately conservative resident-set estimate for
# that footprint. `max_workers` in config overrides the auto-pick (and may use
# every core).
PER_WORKER_GB = 2.5

# Held back by the auto-pick only (#211); an explicit `max_workers` can still use
# every core. (The why — OS/machine headroom — is in the PER_WORKER_GB block.)
RESERVED_CORES = 2

# Caption-pool sizing (#166). Captioning runs after parse as its own concurrent
# stage: many VLM/OCR calls per document, network/IO-bound rather than RAM-bound,
# so it doesn't share parse's RAM-derived auto-formula — a fixed default is more
# honest. 4 is a safe middle ground: a single-GPU local Ollama serializes vision
# requests anyway, while a rate-limited cloud provider tolerates a few in flight.
# Override with `caption_workers` in config.
DEFAULT_CAPTION_WORKERS = 4

# Summarize-pool sizing (#188). Summarization is the heaviest ingest stage (the
# #177 baseline put it at ~59% of wall-clock) and, like captioning, it's a run of
# network/IO-bound LLM calls rather than RAM-bound parse work — so it gets the
# same fixed-default treatment, decoupled from parse-worker sizing. 4 matches
# `caption_workers`: a single-GPU local Ollama serializes anyway, a rate-limited
# cloud provider tolerates a few in flight. Override with `summarize_workers`.
DEFAULT_SUMMARIZE_WORKERS = 4
