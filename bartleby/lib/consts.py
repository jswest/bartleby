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

# Ingest conversion / vision defaults, shared by the `ready` wizard (which
# writes them into config) and `scribe` (which falls back to them when a knob
# is absent from config). Keep the two in lockstep by sourcing both from here.
DEFAULT_PDF_CONVERTER = "pdfplumber"
DEFAULT_HTML_CONVERTER = "docling"
DEFAULT_SPARSE_TEXT_THRESHOLD = 100
DEFAULT_OCR_MIN_CONFIDENCE = 30
DEFAULT_VISION_MAX_DIMENSION = 1024
# VLM image processors (e.g. qwen3-vl) tile images into fixed-size patches and
# crash when an edge is smaller than the patch factor. Images below this on
# either edge — thin rules, banners, sliver crops — are skipped before the VLM
# call. 32 matches qwen3-vl's factor; raise it for models with larger patches.
DEFAULT_VISION_MIN_DIMENSION = 32

# Parse-pool sizing (#165). When `max_workers` is unset, scribe auto-picks
# min(cpu_count, free_ram_gb // PER_WORKER_GB), floored at 1 — so a box that's
# CPU-rich but RAM-poor doesn't launch more parse workers than memory can hold
# and OOM. Each worker loads the embedding model and, for docling ingests, the
# layout/table models; PER_WORKER_GB is a deliberately conservative resident-set
# estimate for that footprint. `max_workers` in config overrides the auto-pick.
PER_WORKER_GB = 2.5
