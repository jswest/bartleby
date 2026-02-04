from pathlib import Path

import apsw
from loguru import logger
import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _migrate_summaries(connection):
    """Migrate old page-level summaries table to document-level if needed."""
    cursor = connection.cursor()

    # Check if summaries table exists and has old schema (summary_id column)
    try:
        cursor.execute("SELECT summary_id FROM summaries LIMIT 0")
    except apsw.SQLError:
        # Table doesn't exist or doesn't have summary_id — no migration needed
        return

    logger.info("Migrating summaries table from page-level to document-level")

    # Delete old summary chunks (raw page text chunks still exist)
    cursor.execute("DELETE FROM chunks WHERE summary_id IS NOT NULL")

    # Drop old page-level summaries table and recreate as document-level
    cursor.execute("DROP TABLE IF EXISTS summaries")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            document_id TEXT PRIMARY KEY REFERENCES documents(document_id),
            title TEXT NOT NULL,
            subtitle TEXT,
            body TEXT NOT NULL
        )
    """)

    # Drop orphaned indexes that reference old schema
    for idx in ("idx_summaries_page_ids", "idx_chunks_summary_order", "uq_chunks_summary_order"):
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {idx}")
        except apsw.SQLError:
            pass

    logger.info("Summaries migration complete")



def create_db(db_dir_path: Path):
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")

    connection = apsw.Connection(str(db_dir_path / "bartleby.db"))

    # Enable extension loading
    connection.enable_load_extension(True)

    # Load sqlite-vec extension
    sqlite_vec.load(connection)

    cursor = connection.cursor()

    # Enable WAL mode for better concurrent write performance
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")

    # Execute all statements - APSW properly handles multiple statements
    # by yielding each one when we iterate through execute()
    for _ in cursor.execute(sql_text):
        pass  # Just consume the iterator

    try:
        cursor.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize')")
    except apsw.Error:
        pass

    connection.close()


def get_connection(db_path: Path):
    connection = apsw.Connection(str(db_path))

    # Set busy timeout to wait up to 30 seconds for locks
    connection.set_busy_timeout(30000)

    # Enable extension loading
    connection.enable_load_extension(True)

    # Load sqlite-vec extension
    sqlite_vec.load(connection)

    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")

    # Migrate old schema before enabling FK enforcement
    _migrate_summaries(connection)
    # Ensure new summaries table exists (for databases created before this schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            document_id TEXT PRIMARY KEY REFERENCES documents(document_id),
            title TEXT NOT NULL,
            subtitle TEXT,
            body TEXT NOT NULL
        )
    """)

    cursor.execute("PRAGMA foreign_keys=ON")

    return connection


