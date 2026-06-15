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
    from bartleby.benchmark.refs import parse_refs
    from bartleby.benchmark.sources import DEFAULT_EXTRACTION

    mod.run(
        _root(args),
        models=parse_refs(args.models),
        documents=_split_csv(args.documents),
        runs=args.runs,
        seed=args.seed,
        ollama_host=args.ollama_host,
        extraction=getattr(args, "extraction", DEFAULT_EXTRACTION) or DEFAULT_EXTRACTION,
    )


def judge(args) -> None:
    from bartleby.benchmark import judging as mod
    from bartleby.benchmark.refs import ModelRef

    mod.run(
        _root(args),
        judge=ModelRef.parse(args.model) if args.model else None,
        passes=args.passes,
    )


def leaderboard(args) -> None:
    import sys
    from pathlib import Path

    from bartleby.benchmark import report
    from bartleby.benchmark.refs import parse_refs
    from bartleby.benchmark.stores import parse_when

    sys.exit(report.leaderboard(
        _root(args),
        models=parse_refs(args.models),
        documents=_split_csv(args.documents),
        judges=parse_refs(args.judges),
        extractions=_split_csv(getattr(args, "extractions", None)),
        since=parse_when(args.since) if args.since else None,
        until=parse_when(args.until, end=True) if args.until else None,
        output=Path(args.output) if args.output else None,
        min_schema=args.min_schema,
    ))


def blind(args) -> None:
    import sys
    from pathlib import Path

    from bartleby.benchmark import report
    from bartleby.benchmark.refs import parse_refs

    root = _root(args)
    sys.exit(report.blind(
        root,
        out_dir=Path(args.out) if args.out else root.root / "blind",
        seed=args.seed,
        models=parse_refs(args.models),
        documents=_split_csv(args.documents),
    ))


def errors(args) -> None:
    import sys

    from bartleby.benchmark import report
    from bartleby.benchmark.refs import parse_refs

    sys.exit(report.errors(
        _root(args),
        models=parse_refs(args.models),
        documents=_split_csv(args.documents),
    ))
