import argparse
import sys

# Cheap, dependency-free constants — safe to import at module load (and thus at
# parser-build time) without pulling the provider package's pydantic cost into
# every `bartleby` invocation. These keep scribe's `choices=` in lockstep with
# what the config wizard accepts.
from bartleby.lib.consts import (
    ALLOWED_HTML_CONVERTERS,
    ALLOWED_PDF_CONVERTERS,
    ALLOWED_PROVIDERS,
)


def main():
    # `bartleby skill <name> [args]` bypasses argparse so we can pass the
    # rest of argv straight through to the skill script (its own argparse
    # parses --project, --limit, etc. without `bartleby`'s top-level parser
    # rejecting them).
    if len(sys.argv) >= 2 and sys.argv[1] == "skill":
        from bartleby.commands.skill import dispatch
        dispatch(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        prog="bartleby",
        description="Bartleby, the Scrivener - A document analysis toolkit that might prefer not to."
    )
    from bartleby import __version__
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("config", help="Configure Bartleby settings")

    ready_parser = subparsers.add_parser(
        "ready",
        help="Install or refresh the Bartleby skill into your agent harness",
    )
    ready_parser.add_argument(
        "--check", action="store_true",
        help="Report whether the installed skill is up to date and exit "
             "non-zero if it's missing or stale; write nothing.",
    )
    ready_parser.add_argument(
        "--force", action="store_true",
        help="Reinstall even if the installed skill is already up to date.",
    )
    ready_parser.add_argument(
        "--dest", type=str, default=None,
        help="Skill directory to (re)create "
             "(default: ~/.claude/skills/bartleby).",
    )

    project_parser = subparsers.add_parser("project", help="Manage projects")
    project_sub = project_parser.add_subparsers(dest="project_command")
    pc = project_sub.add_parser("create", help="Create a new project")
    pc.add_argument("name", type=str)
    project_sub.add_parser("list", help="List all projects")
    pu = project_sub.add_parser("use", help="Switch to an existing project")
    pu.add_argument("name", type=str)
    pi = project_sub.add_parser("info", help="Show project details")
    pi.add_argument("name", type=str, nargs="?", default=None)
    pi.add_argument(
        "--verify",
        action="store_true",
        help="Run read-only corpus integrity checks; exit non-zero on any failure",
    )
    pd = project_sub.add_parser("delete", help="Delete a project")
    pd.add_argument("name", type=str)
    pd.add_argument("-y", "--yes", action="store_true")
    pup = project_sub.add_parser(
        "upgrade",
        help="Apply additive schema upgrades to bring a project up to date",
    )
    pup.add_argument("name", type=str)
    ppub = project_sub.add_parser(
        "publish",
        help="Publish a findings-free copy of a corpus (+ originals) to S3",
    )
    ppub.add_argument("name", type=str)
    ppub.add_argument(
        "--to", required=True, metavar="S3_URL",
        help="Destination S3 URL, e.g. s3://my-bucket/corpora/acme",
    )
    pimp = project_sub.add_parser(
        "import",
        help="Import a published corpus from S3 as a new local project",
    )
    pimp.add_argument("name", type=str)
    pimp.add_argument(
        "--from", required=True, metavar="S3_URL", dest="from_url",
        help="Source S3 URL, e.g. s3://my-bucket/corpora/acme",
    )
    pimp.add_argument(
        "--without-tags", action="store_true",
        help="Drop tag definitions and assignments from the imported corpus",
    )
    pimp.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing project of the same name "
             "(drops its local findings)",
    )

    scribe_parser = subparsers.add_parser(
        "scribe",
        help="Ingest PDF, HTML, MD, TXT, and image files into a project",
    )
    scribe_parser.add_argument(
        "--files", required=True, type=str, nargs="+", metavar="PATH",
        help="One or more files and/or directories to ingest. Directories are "
             "walked recursively; a file reachable from more than one path is "
             "ingested once.",
    )
    scribe_parser.add_argument(
        "--only", type=str, action="append", default=None, metavar="TYPE",
        help="Restrict ingestion to the given file type(s): pdf, html, md, txt, "
             "image. Repeatable and/or comma-separated (e.g. --only pdf,html). "
             "Filters on the resolved type, so content-sniffed files are "
             "included.",
    )
    scribe_parser.add_argument("--project", type=str, default=None)
    scribe_parser.add_argument("--model", type=str, default=None)
    scribe_parser.add_argument(
        "--provider", type=str,
        choices=ALLOWED_PROVIDERS, default=None,
    )
    scribe_parser.add_argument(
        "--pdf-converter", type=str,
        choices=ALLOWED_PDF_CONVERTERS, default=None,
        help="PDF converter; overrides pdf_converter in ~/.bartleby/config.yaml.",
    )
    scribe_parser.add_argument(
        "--html-converter", type=str,
        choices=ALLOWED_HTML_CONVERTERS, default=None,
        help=(
            "HTML converter; overrides html_converter in ~/.bartleby/config.yaml. "
            "'sec2md' routes iXBRL EDGAR filings to sec2md and other HTML to docling."
        ),
    )
    scribe_parser.add_argument("--verbose", action="store_true")
    scribe_parser.add_argument(
        "--timings", action="store_true",
        help="Benchmark mode: time each document's parse/embed/caption/summarize "
             "stages, print per-doc wall-clock to stderr and an aggregate "
             "(docs/sec, pages/sec, per-stage breakdown) as JSON to stdout. "
             "Skips already-ingested files, so run against a fresh project.",
    )

    session_parser = subparsers.add_parser("session", help="Manage agent sessions")
    session_sub = session_parser.add_subparsers(dest="session_command")
    ss = session_sub.add_parser("start", help="Start a new session and mark it active")
    ss.add_argument("--no-memory", action="store_true")
    ss.add_argument(
        "--harness", type=str, default=None,
        help="Harness that will author this session's findings (e.g. claude-code). "
             "Auto-detected when omitted.",
    )
    ss.add_argument(
        "--model", type=str, default=None,
        help="Model that will author this session's findings (e.g. claude-opus-4-8).",
    )
    ss.add_argument("--project", type=str, default=None)
    sc = session_sub.add_parser("current", help="Show the active session")
    sc.add_argument("--project", type=str, default=None)
    se = session_sub.add_parser("end", help="End the active session")
    se.add_argument("--project", type=str, default=None)
    sset = session_sub.add_parser(
        "set", help="Set the active session's model and/or harness after the fact"
    )
    sset.add_argument("--harness", type=str, default=None)
    sset.add_argument("--model", type=str, default=None)
    sset.add_argument("--project", type=str, default=None)

    embed_parser = subparsers.add_parser(
        "embed", help="Print the BGE embedding for a string as a JSON array to stdout"
    )
    embed_parser.add_argument("text", type=str)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark summarizer models against the committed corpus"
    )
    benchmark_sub = benchmark_parser.add_subparsers(dest="benchmark_command")
    bs = benchmark_sub.add_parser(
        "summarize",
        help="Append summarize runs for every model × corpus doc (or a subset)",
    )
    bs.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated <provider>/<model> list (e.g. ollama/gemma4:e2b); "
             "default: every model in models.yaml.",
    )
    bs.add_argument(
        "--documents", type=str, default=None,
        help="Comma-separated doc-id list; default: every doc in corpus.yaml.",
    )
    bs.add_argument(
        "--runs", type=int, default=1,
        help="Runs to append per (model, doc) cell this invocation (default 1).",
    )
    bs.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the matrix-order shuffle so the plan is reproducible.",
    )
    bs.add_argument(
        "--ollama-host", type=str, default=None,
        help="Override the Ollama host (default http://localhost:11434).",
    )
    bj = benchmark_sub.add_parser(
        "judge",
        help="Top up blind judge scores for every distinct summary on record",
    )
    bj.add_argument(
        "--model", type=str, default=None,
        help="Judge as <provider>/<model>; default: the first entry in judges.yaml.",
    )
    bj.add_argument(
        "--passes", type=int, default=3,
        help="Judgments each distinct summary should have (default 3). "
             "Declarative and idempotent: re-running the same value is a "
             "no-op; want more judgments on what's already judged, raise it "
             "(--passes 8 adds 5 to a summary that has 3).",
    )
    bl = benchmark_sub.add_parser(
        "leaderboard", help="Ranked report merging runs, schema gate, and judge scores"
    )
    bl.add_argument(
        "--judges", type=str, default=None,
        help="Comma-separated <provider>/<model> judge filter; default: all on record.",
    )
    bl.add_argument(
        "--since", type=str, default=None,
        help="Only records on/after this ISO date (e.g. 2026-06-01).",
    )
    bl.add_argument(
        "--until", type=str, default=None,
        help="Only records on/before this ISO date.",
    )
    bl.add_argument(
        "--output", type=str, default=None,
        help="Also write the leaderboard to this CSV file.",
    )
    bl.add_argument(
        "--min-schema", type=float, default=100.0,
        help="Minimum schema-valid %% to survive the hard gate (default 100).",
    )

    bb = benchmark_sub.add_parser(
        "blind", help="Write blinded summaries + a key file for a human spot-check"
    )
    bb.add_argument(
        "--out", type=str, default=None,
        help="Output directory (default <benchmarks-dir>/blind).",
    )
    bb.add_argument(
        "--seed", type=int, default=None,
        help="Seed the per-model pick + label shuffle for reproducibility.",
    )

    be = benchmark_sub.add_parser("errors", help="List failed runs")

    for sub in (bs, bj, bl, bb, be):
        sub.add_argument(
            "--benchmarks-dir", type=str, default="benchmarks",
            help="Benchmarks directory (default ./benchmarks).",
        )
    for sub in (bl, bb, be):
        sub.add_argument(
            "--models", type=str, default=None,
            help="Comma-separated <provider>/<model> filter.",
        )
        sub.add_argument(
            "--documents", type=str, default=None,
            help="Comma-separated doc-id filter.",
        )

    logs_parser = subparsers.add_parser("logs", help="View the audit log for a session")
    logs_parser.add_argument("--session", type=str, default=None)
    logs_parser.add_argument("--limit", type=int, default=50)
    logs_parser.add_argument("--project", type=str, default=None)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch a local SvelteKit UI for browsing the active project's findings.",
    )
    serve_parser.add_argument(
        "--project", type=str, default=None,
        help="Browse this project instead of the active one, for this server "
             "only — does not change the persisted active project.",
    )

    # Stub so `bartleby --help` lists `skill`; the real dispatch is above.
    subparsers.add_parser(
        "skill",
        help="Run a skill script (use `bartleby skill --help` to list).",
        add_help=False,
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatchers = {
        "config": lambda: _config(),
        "ready": lambda: _ready(args),
        "project": lambda: _project(args, project_parser),
        "scribe": lambda: _scribe(args),
        "session": lambda: _session(args, session_parser),
        "embed": lambda: _embed(args),
        "benchmark": lambda: _benchmark(args, benchmark_parser),
        "logs": lambda: _logs(args),
        "serve": lambda: _serve(args),
    }
    dispatchers[args.command]()


def _config():
    from bartleby.commands.config import main as config_main
    from bartleby.lib import console

    console.splash()
    config_main()


def _ready(args):
    from pathlib import Path

    from bartleby.commands.ready import main as ready_main

    dest = Path(args.dest).expanduser() if args.dest else None
    ready_main(dest=dest, check=args.check, force=args.force)


def _scribe(args):
    from bartleby.commands.scribe import main as scribe_main
    from bartleby.lib import console

    console.splash()
    # Scribe's expected failure modes — no active project (RuntimeError), an
    # invalid configured converter or a typo'd --only (ValueError), a missing
    # path (FileNotFoundError) — are user errors, not bugs. Surface them as a
    # one-line message on stderr and exit 1 instead of dumping a traceback.
    try:
        scribe_main(
            project=args.project,
            files=args.files,
            only=args.only,
            model=args.model,
            provider=args.provider,
            pdf_converter=args.pdf_converter,
            html_converter=args.html_converter,
            verbose=args.verbose,
            timings=args.timings,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        console.error(str(e))
        sys.exit(1)


def _embed(args):
    from bartleby.commands.embed import main as embed_main
    embed_main(args.text)


def _benchmark(args, parser):
    from bartleby.commands import benchmark as benchmark_cmd

    if not args.benchmark_command:
        parser.print_help()
        sys.exit(1)

    getattr(benchmark_cmd, args.benchmark_command)(args)


def _logs(args):
    from bartleby.commands.logs import main as logs_main
    logs_main(session=args.session, limit=args.limit, project=args.project)


def _serve(args):
    from bartleby.commands.serve import main as serve_main
    serve_main(project=args.project)


def _project(args, parser):
    from bartleby.commands import project as project_cmd

    if not args.project_command:
        parser.print_help()
        sys.exit(1)

    if args.project_command == "create":
        project_cmd.create(name=args.name)
    elif args.project_command == "list":
        project_cmd.list_()
    elif args.project_command == "use":
        project_cmd.use(name=args.name)
    elif args.project_command == "info":
        project_cmd.info(name=args.name, verify=args.verify)
    elif args.project_command == "delete":
        project_cmd.delete(name=args.name, yes=args.yes)
    elif args.project_command == "upgrade":
        project_cmd.upgrade(name=args.name)
    elif args.project_command == "publish":
        project_cmd.publish(name=args.name, to=args.to)
    elif args.project_command == "import":
        project_cmd.import_(
            name=args.name, from_url=args.from_url,
            without_tags=args.without_tags, force=args.force,
        )


def _session(args, parser):
    from bartleby.commands import session as session_cmd

    if not args.session_command:
        parser.print_help()
        sys.exit(1)

    project = getattr(args, "project", None)
    if args.session_command == "start":
        session_cmd.start(
            project=project, no_memory=args.no_memory,
            harness=args.harness, model=args.model,
        )
    elif args.session_command == "current":
        session_cmd.current(project=project)
    elif args.session_command == "end":
        session_cmd.end(project=project)
    elif args.session_command == "set":
        session_cmd.set_provenance(
            project=project, harness=args.harness, model=args.model,
        )


if __name__ == "__main__":
    main()
