"""Provider client construction shared by summarize and judge."""

from __future__ import annotations

from bartleby.benchmark.stores import BenchmarkRoot


def make_openai_client(root: BenchmarkRoot):
    import os

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=root.root / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY not set (looked in env and {root.root / '.env'})")
    from openai import OpenAI
    return OpenAI()
