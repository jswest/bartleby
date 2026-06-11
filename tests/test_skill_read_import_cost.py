"""Read-only skill scripts must not import the embedding/model stack (#371).

``scan``, ``list_documents``, ``read_chunks``, and ``describe_corpus`` are
FTS-only read paths. They share ``_common`` / ``_tags``, which used to import
the chunker + ``embed_texts`` at module top — dragging ``docling`` / ``filetype``
(and the lazy ``sentence_transformers`` model loader) into every invocation. The
heavy imports are now function-local to the helpers that actually embed
(``embed_body_chunks`` / ``find_similar_tag``), so a read-script *import* never
pays for them. Each assertion runs in a fresh interpreter so a module another
test already imported can't mask a regression.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

READ_SCRIPTS = ("scan", "list_documents", "read_chunks", "describe_corpus")

# The embedding/model stack the read path must never pull in at import. (numpy
# is intentionally excluded: it rides in via sqlite-vec on the DB connection,
# which is core read-path infrastructure, not the embedding stack.)
FORBIDDEN = ("bartleby.ingest.embed", "sentence_transformers", "torch",
             "docling", "filetype", "bartleby.ingest.chunk")


@pytest.mark.parametrize("script", READ_SCRIPTS)
def test_read_script_import_skips_embedding_stack(script: str) -> None:
    forbidden = ",".join(repr(m) for m in FORBIDDEN)
    code = (
        "import sys\n"
        f"import bartleby.skill_scripts.{script}\n"
        f"forbidden = ({forbidden},)\n"
        "leaked = [m for m in forbidden "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "assert not leaked, leaked\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"importing {script} leaked heavy modules:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
