from pathlib import Path

import apsw
import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


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
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")

    return connection


