"""Constants used across the v1 ingest pipeline.

Other knobs (chunker limits, search defaults) live in the module that owns
them — keep this file focused on shared values.
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

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
