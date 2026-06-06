import argparse
import sys


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
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("ready", help="Configure Bartleby settings")

    project_parser = subparsers.add_parser("project", help="Manage projects")
    project_sub = project_parser.add_subparsers(dest="project_command")
    pc = project_sub.add_parser("create", help="Create a new project")
    pc.add_argument("name", type=str)
    project_sub.add_parser("list", help="List all projects")
    pu = project_sub.add_parser("use", help="Switch to an existing project")
    pu.add_argument("name", type=str)
    pi = project_sub.add_parser("info", help="Show project details")
    pi.add_argument("name", type=str, nargs="?", default=None)
    pd = project_sub.add_parser("delete", help="Delete a project")
    pd.add_argument("name", type=str)
    pd.add_argument("-y", "--yes", action="store_true")
    pup = project_sub.add_parser(
        "upgrade",
        help="Apply additive schema upgrades to bring a project up to date",
    )
    pup.add_argument("name", type=str)

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
        choices=["anthropic", "openai", "ollama"], default=None,
    )
    scribe_parser.add_argument(
        "--pdf-converter", type=str,
        choices=["pdfplumber", "docling"], default=None,
        help="PDF converter; overrides pdf_converter in ~/.bartleby/config.yaml.",
    )
    scribe_parser.add_argument(
        "--html-converter", type=str,
        choices=["docling", "sec2md"], default=None,
        help=(
            "HTML converter; overrides html_converter in ~/.bartleby/config.yaml. "
            "'sec2md' routes iXBRL EDGAR filings to sec2md and other HTML to docling."
        ),
    )
    scribe_parser.add_argument("--verbose", action="store_true")

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

    logs_parser = subparsers.add_parser("logs", help="View the audit log for a session")
    logs_parser.add_argument("--session", type=str, default=None)
    logs_parser.add_argument("--limit", type=int, default=50)
    logs_parser.add_argument("--project", type=str, default=None)

    subparsers.add_parser(
        "serve",
        help="Launch a local SvelteKit UI for browsing the active project's findings.",
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
        "ready": lambda: _ready(),
        "project": lambda: _project(args, project_parser),
        "scribe": lambda: _scribe(args),
        "session": lambda: _session(args, session_parser),
        "embed": lambda: _embed(args),
        "logs": lambda: _logs(args),
        "serve": lambda: _serve(),
    }
    dispatchers[args.command]()


def _ready():
    from bartleby.commands.ready import main as ready_main
    from bartleby.lib import console

    console.splash()
    ready_main()


def _scribe(args):
    from bartleby.commands.scribe import main as scribe_main
    from bartleby.lib import console

    console.splash()
    scribe_main(
        project=args.project,
        files=args.files,
        only=args.only,
        model=args.model,
        provider=args.provider,
        pdf_converter=args.pdf_converter,
        html_converter=args.html_converter,
        verbose=args.verbose,
    )


def _embed(args):
    from bartleby.commands.embed import main as embed_main
    embed_main(args.text)


def _logs(args):
    from bartleby.commands.logs import main as logs_main
    logs_main(session=args.session, limit=args.limit, project=args.project)


def _serve():
    from bartleby.commands.serve import main as serve_main
    serve_main()


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
        project_cmd.info(name=args.name)
    elif args.project_command == "delete":
        project_cmd.delete(name=args.name, yes=args.yes)
    elif args.project_command == "upgrade":
        project_cmd.upgrade(name=args.name)


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
