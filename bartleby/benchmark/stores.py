"""Append-only per-cell JSONL stores under the benchmarks directory.

Layout (relative to the benchmarks root, default ``benchmarks/`` in the CWD):

    corpus.yaml   models.yaml   judges.yaml      # tracked configuration
    corpus/                                      # tracked PDFs
    sources/<doc-id>.txt                         # ignored: extracted-text cache (pdfplumber)
    sources/<doc-id>-<extraction>.txt            # ignored: extraction-variant cache
    results/<provider>_<model>_<doc-id>.jsonl           # ignored: run records (pdfplumber)
    results/<provider>_<model>_<doc-id>_x-<extraction>.jsonl  # ignored: named-extraction runs
    judgements/<provider>_<model>_<doc-id>_<judge-provider>_<judge-model>.jsonl
    judgements/<provider>_<model>_<doc-id>_x-<extraction>_<judge-provider>_<judge-model>.jsonl

Filenames are write-side organization only — every record carries its own
identity fields (``provider``, ``model``, ``doc``, ``extraction``, and for
judgments the judge pair), so readers glob and filter on record contents and
never parse names.

A cell is (provider, model, doc, extraction). The default extraction is
``"pdfplumber"`` and its files omit the ``_x-<extraction>`` segment so
existing stores are backward-compatible.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from bartleby.benchmark.refs import ModelRef

DEFAULT_ROOT = Path("benchmarks")


class BenchmarkRoot:
    """Path arithmetic for one benchmarks directory."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    def require(self) -> "BenchmarkRoot":
        if not (self.root / "corpus.yaml").exists():
            raise SystemExit(
                f"No corpus.yaml under {self.root}/ — run from the bartleby "
                f"repo root or pass --benchmarks-dir"
            )
        return self

    @property
    def corpus_yaml(self) -> Path:
        return self.root / "corpus.yaml"

    @property
    def models_yaml(self) -> Path:
        return self.root / "models.yaml"

    @property
    def judges_yaml(self) -> Path:
        return self.root / "judges.yaml"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def judgements_dir(self) -> Path:
        return self.root / "judgements"

    def result_path(self, ref: ModelRef, doc_id: str,
                    extraction: str = "pdfplumber") -> Path:
        suffix = f"_x-{extraction}" if extraction != "pdfplumber" else ""
        return self.results_dir / f"{ref.slug}_{doc_id}{suffix}.jsonl"

    def judgement_path(self, ref: ModelRef, doc_id: str, judge: ModelRef,
                       extraction: str = "pdfplumber") -> Path:
        suffix = f"_x-{extraction}" if extraction != "pdfplumber" else ""
        return self.judgements_dir / f"{ref.slug}_{doc_id}{suffix}_{judge.slug}.jsonl"

    def source_path(self, doc_id: str, extraction: str = "pdfplumber") -> Path:
        suffix = f"-{extraction}" if extraction != "pdfplumber" else ""
        return self.sources_dir / f"{doc_id}{suffix}.txt"

    def load_models(self) -> list[ModelRef]:
        return [ModelRef.parse(m) for m in self._yaml_list(self.models_yaml, "models")]

    def load_judges(self) -> list[ModelRef]:
        return [ModelRef.parse(m) for m in self._yaml_list(self.judges_yaml, "judges")]

    @staticmethod
    def _yaml_list(path: Path, key: str) -> list:
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        entries = (yaml.safe_load(path.read_text()) or {}).get(key) or []
        if not isinstance(entries, list):
            raise SystemExit(f"`{key}:` in {path} must be a list")
        if not entries:
            raise SystemExit(f"No entries under `{key}:` in {path}")
        return entries


def make_openai_client(root: BenchmarkRoot):
    """Construct an OpenAI client, loading the key from env or ``root/.env``.

    Imports of ``dotenv``/``openai`` stay function-local so ``bartleby --help``
    (and any non-OpenAI benchmark path) never pays for them.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=root.root / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY not set (looked in env and {root.root / '.env'})")
    from openai import OpenAI
    return OpenAI()


def append_record(path: Path, record: dict) -> None:
    """Append one timestamped JSON line; creates the store on first write."""
    record = {"timestamp": time.time(), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path: Path) -> list[dict]:
    """Tolerant reader: a malformed line (interrupted append) is skipped."""
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping malformed line in {path}", file=sys.stderr)
    return records


def read_store(dir_path: Path) -> list[dict]:
    """All records across a store directory, in filename-then-line order."""
    records: list[dict] = []
    for path in sorted(dir_path.glob("*.jsonl")):
        records.extend(read_records(path))
    return records


def parse_when(raw: str, *, end: bool = False) -> float:
    """ISO date/datetime → epoch seconds (local time). A date-only ``--until``
    means "through that whole day", so ``end=True`` rolls it to the next
    midnight and callers compare with ``<``."""
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        raise SystemExit(f"Can't parse date {raw!r} (expected ISO, e.g. 2026-06-09)")
    if end and len(raw) == 10:
        dt += timedelta(days=1)
    return dt.timestamp()


def in_window(record: dict, since: float | None, until: float | None) -> bool:
    ts = record.get("timestamp")
    if ts is None:
        return False
    if since is not None and ts < since:
        return False
    if until is not None and ts >= until:
        return False
    return True
