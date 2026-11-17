"""SQLite connection management and WAL mode configuration."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class Database:
    """Manages SQLite connection lifecycle and Pragmas."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection with WAL mode enabled."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding an active connection, auto-committing on success."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
