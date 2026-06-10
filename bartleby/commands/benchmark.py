"""CLI handlers for the `bartleby benchmark` command group.

Thin argument plumbing only — the logic lives in ``bartleby.benchmark.*``,
imported lazily so `bartleby --help` never pays for rich/tiktoken/openai.
"""

from __future__ import annotations


def _split_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("Empty comma-separated list")
    return parts


def _root(args):
    from bartleby.benchmark.stores import BenchmarkRoot

    return BenchmarkRoot(args.benchmarks_dir)


def summarize(args) -> None:
    from bartleby.benchmark import summarize as mod
    from bartleby.benchmark.refs import parse_flag_refs

    mod.run(
        _root(args),
        models=parse_flag_refs(args.models),
        documents=_split_csv(args.documents),
        runs=args.runs,
        seed=args.seed,
        ollama_host=args.ollama_host,
    )


def judge(args) -> None:
    from bartleby.benchmark import judging as mod
    from bartleby.benchmark.refs import ModelRef

    mod.run(
        _root(args),
        judge=ModelRef.from_flag(args.model) if args.model else None,
        passes=args.passes,
    )
