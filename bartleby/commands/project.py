"""`bartleby project` — create / list / use / info / delete / upgrade."""

from __future__ import annotations

import sys

import apsw
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from bartleby.db.connection import _attach, project_db_path
from bartleby.db.schema import SCHEMA_VERSION
from bartleby.db import upgrades as upgrades_mod
from bartleby.integrity import run_all_checks
from bartleby.lib import console
from bartleby.project import (
    create_project,
    delete_project,
    get_active_project,
    get_project_info,
    list_projects,
    set_active_project,
    validate_project_name,
)


_console = Console()


def create(*, name: str) -> None:
    try:
        project_dir = create_project(name)
    except (ValueError, FileExistsError) as e:
        console.error(str(e))
        sys.exit(1)
    _console.print(f"[bold green]Created project '{name}'[/bold green]")
    _console.print(f"Location: [cyan]{project_dir}[/cyan]")
    _console.print(f"Active project set to: [bold]{name}[/bold]")


def list_(*, _: None = None) -> None:
    projects = list_projects()
    if not projects:
        _console.print(
            "No projects found. Create one with `bartleby project create <name>`"
        )
        return

    table = Table(title="Projects")
    table.add_column("", width=2)
    table.add_column("Name", style="bold")
    table.add_column("Database")
    for p in projects:
        marker = "*" if p["is_active"] else ""
        db_status = "[green]ready[/green]" if p["has_db"] else "[red]no db[/red]"
        table.add_row(marker, p["name"], db_status)
    _console.print(table)


def use(*, name: str) -> None:
    try:
        set_active_project(name)
    except (ValueError, FileNotFoundError) as e:
        console.error(str(e))
        sys.exit(1)
    _console.print(f"Active project set to: [bold]{name}[/bold]")


def info(*, name: str | None, verify: bool = False) -> None:
    name = name or get_active_project()
    if not name:
        console.error(
            "No active project. Specify a name: `bartleby project info <name>`"
        )
        sys.exit(1)
    try:
        i = get_project_info(name)
    except (ValueError, FileNotFoundError) as e:
        console.error(str(e))
        sys.exit(1)
    except RuntimeError as e:
        # A schema-version mismatch makes get_project_info's open_db raise. Under
        # --verify that IS the finding (the integrity audit names it), so press
        # on to the checks instead of aborting; otherwise it's fatal as before.
        if not verify:
            console.error(str(e))
            sys.exit(1)
        i = None

    if i is not None:
        table = Table(title=f"Project: {i['name']}")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Active", "yes" if i["is_active"] else "no")
        table.add_row("Path", str(i["path"]))
        table.add_row("Database", "ready" if i["has_db"] else "missing")
        if i["has_db"]:
            table.add_row("DB size", f"{i['db_size_mb']} MB")
            table.add_row("Schema version", str(i["schema_version"]))
            table.add_row("Embedding model", str(i["embedding_model"]))
            table.add_row("Documents", str(i["document_count"]))
            table.add_row("Sessions", str(i["session_count"]))
            table.add_row("Findings", str(i["finding_count"]))
            c = i["chunk_counts"]
            table.add_row(
                "Chunks",
                f"{c['document']} document  /  {c['image']} image  /  "
                f"{c['summary']} summary  /  {c['finding']} finding",
            )
            f = i["failed_ingests"]
            if f["total"]:
                capped_note = (
                    f" ({f['capped']} capped, not retried)" if f["capped"] else ""
                )
                table.add_row(
                    "Failed units",
                    f"[yellow]{f['total']} incomplete{capped_note}[/yellow]",
                )
        _console.print(table)

    if verify:
        _verify(name)


def _verify(name: str) -> None:
    """Run the read-only integrity audit and exit non-zero on any failure.

    Opens its own raw connection (vec extension attached, no schema-version gate)
    so a stamped-vs-code mismatch is *reported* by the schema check rather than
    raised before any check runs. Read-only throughout.
    """
    db_path = project_db_path(name)
    if not db_path.exists():
        console.error(f"Project '{name}' has no database to verify.")
        sys.exit(1)

    conn = apsw.Connection(str(db_path))
    try:
        _attach(conn)
        results = run_all_checks(conn)
    finally:
        conn.close()

    _console.print("\n[bold]Integrity checks[/bold]")
    for r in results:
        mark = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        _console.print(f"  {mark}  {r.name}: {r.detail}")

    if not all(r.passed for r in results):
        sys.exit(1)


def upgrade(*, name: str) -> None:
    """Apply additive schema upgrades to bring a project DB up to ``SCHEMA_VERSION``.

    Bypasses ``open_db``'s strict version check (which would refuse a stale
    DB outright). Non-additive bumps raise — re-ingest is the only path.
    """
    try:
        validate_project_name(name)
    except ValueError as e:
        console.error(str(e))
        sys.exit(1)
    db_path = project_db_path(name)
    if not db_path.exists():
        console.error(f"Project '{name}' has no database.")
        sys.exit(1)

    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        row = cur.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            console.error("Database has no schema_version. Recreate the project.")
            sys.exit(1)
        current = int(row[0])
        if current > SCHEMA_VERSION:
            console.error(
                f"Database is at v{current}, newer than code's "
                f"v{SCHEMA_VERSION}. Update the code, not the DB."
            )
            sys.exit(1)

        # Always call upgrades.upgrade — even when already at SCHEMA_VERSION. The
        # version-step loop won't run, but the always-runs tail (idempotent
        # `INSERT OR IGNORE embedding_model` + `upgraded_at`) still does. A v9
        # corpus created before #517 has no embedding_model key, and this is the
        # only CLI path that backfills it; skipping the call here (the old bug)
        # left those corpora forever unable to publish an importable artifact.
        try:
            upgrades_mod.upgrade(conn, current)
        except (RuntimeError, apsw.Error) as e:
            console.error(f"Upgrade failed: {e}")
            sys.exit(1)
    finally:
        conn.close()

    if current == SCHEMA_VERSION:
        _console.print(
            f"Project '{name}' is already at schema v{SCHEMA_VERSION}; "
            "ensured metadata is current."
        )
    else:
        _console.print(
            f"[bold green]Upgraded '{name}'[/bold green] "
            f"from v{current} to v{SCHEMA_VERSION}."
        )


def publish(*, name: str, to: str) -> None:
    """Publish a findings-free copy of a corpus (+ originals) to an S3 URL.

    Writes a ``VACUUM INTO`` snapshot of the corpus DB, strips the session layer
    on that copy, gathers the original files content-addressed by ``file_hash``,
    and uploads the ``.db`` + files to ``--to``. The source corpus is never
    mutated.
    """
    from botocore.exceptions import (
        ClientError,
        EndpointConnectionError,
        NoCredentialsError,
    )

    from bartleby.share.publish import publish_project

    try:
        result = publish_project(name, to)
    except (ValueError, FileNotFoundError) as e:
        console.error(str(e))
        sys.exit(1)
    except (ClientError, NoCredentialsError, EndpointConnectionError) as e:
        # A bad bucket/URL, missing/expired credentials, or an unreachable
        # endpoint should be a clean error, not a boto3 traceback.
        console.error(f"S3 error: {e}")
        sys.exit(1)

    _console.print(
        f"[bold green]Published '{name}'[/bold green] to "
        f"[cyan]{result['destination']}[/cyan]"
    )
    _console.print(f"Database: [cyan]{result['db_url']}[/cyan]")
    _console.print(f"Files: {result['file_count']} uploaded (keyed by file_hash)")


def import_(*, name: str, from_url: str, without_tags: bool) -> None:
    """Import a published corpus from an S3 URL as a fresh local project.

    Downloads the published ``.db`` + originals, verifies schema and embedding
    model BEFORE trusting the corpus (a mismatch — or a missing embedding-model
    key — is a hard refuse, no ``--force``), then adopts the ``.db`` as-is and
    rewrites local file paths by ``file_hash``. ``--without-tags`` drops the
    adopted tag definitions and assignments.
    """
    from botocore.exceptions import (
        ClientError,
        EndpointConnectionError,
        NoCredentialsError,
    )

    from bartleby.share.import_ import ImportRefused, import_project

    try:
        result = import_project(name, from_url, without_tags=without_tags)
    except (ValueError, ImportRefused) as e:
        console.error(str(e))
        sys.exit(1)
    except (ClientError, NoCredentialsError, EndpointConnectionError) as e:
        # A wrong URL, a missing bartleby.db at the prefix, missing/expired
        # credentials, or an unreachable endpoint should be a clean error, not a
        # boto3 traceback. import_project already tore down any half-built project.
        console.error(f"S3 error: {e}")
        sys.exit(1)

    _console.print(
        f"[bold green]Imported '{result['project']}'[/bold green] from "
        f"[cyan]{result['source']}[/cyan]"
    )
    _console.print(
        f"Files: {result['file_count']} landed (keyed by file_hash)"
    )
    if result["tags_dropped"]:
        _console.print("[yellow]Tags dropped (--without-tags)[/yellow]")
    _console.print(f"Active project set to: [bold]{result['project']}[/bold]")


def delete(*, name: str, yes: bool) -> None:
    if not yes:
        if not Confirm.ask(
            f"Delete project '{name}' and all its data?", default=False
        ):
            _console.print("Cancelled.")
            return
    try:
        delete_project(name)
    except (ValueError, FileNotFoundError) as e:
        console.error(str(e))
        sys.exit(1)
    _console.print(f"[bold red]Deleted project '{name}'[/bold red]")
    if get_active_project() is None:
        _console.print(
            "[yellow]No active project. Set one with `bartleby project use <name>`[/yellow]"
        )
