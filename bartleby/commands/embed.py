"""`bartleby embed <text>` — print the BGE embedding as a JSON array to stdout.

Used by the skill's ``search`` script for semantic queries. The skill MUST
invoke this with list-form ``subprocess.run(["bartleby", "embed", query])``;
never ``shell=True``. With list-form the query is ``argv[1]`` and the shell
never sees it — no escaping or sanitization needed (SPEC §5.5).
"""

from __future__ import annotations

import json
import sys

from bartleby.ingest.embed import embed_texts


def main(text: str) -> None:
    if not text or not text.strip():
        print("error: embed text must be non-empty", file=sys.stderr)
        sys.exit(1)

    [vector] = embed_texts([text])
    json.dump(vector, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
