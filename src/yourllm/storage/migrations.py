"""Lightweight, versioned SQLite schema migration runner."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

# List of migration functions: (version, description, migration_callable)
MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = []


def migration(version: int, description: str) -> Callable[[Callable[[sqlite3.Connection], None]], Callable[[sqlite3.Connection], None]]:
    """Decorator to register a numbered schema migration."""
    def decorator(func: Callable[[sqlite3.Connection], None]) -> Callable[[sqlite3.Connection], None]:
        MIGRATIONS.append((version, description, func))
        return func
    return decorator


@migration(1, "Initial schema: api_keys and usage_stats")
def migration_v1(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            status TEXT NOT NULL,
            revoked_at TEXT
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_stats (
            model TEXT NOT NULL,
            period_start TEXT NOT NULL,
            total_requests INTEGER NOT NULL DEFAULT 0,
            successful_requests INTEGER NOT NULL DEFAULT 0,
            failed_requests INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_latency_ms REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (model, period_start)
        );
    """)


class MigrationRunner:
    """Discovers and applies pending database migrations in transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_migration_table(self) -> None:
        """Create schema_migrations tracking table if absent."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
        """)

    def get_applied_versions(self) -> set[int]:
        """Return the set of already applied migration version numbers."""
        self.ensure_migration_table()
        cur = self.conn.execute("SELECT version FROM schema_migrations;")
        return {row[0] for row in cur.fetchall()}

    def apply_pending(self) -> int:
        """Apply all registered migrations that have not yet been run."""
        self.ensure_migration_table()
        applied = self.get_applied_versions()
        count = 0

        # Sort migrations strictly by version ascending
        sorted_migrations = sorted(MIGRATIONS, key=lambda m: m[0])
        for version, desc, func in sorted_migrations:
            if version not in applied:
                func(self.conn)
                now = datetime.now(UTC).isoformat()
                self.conn.execute(
                    "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?);",
                    (version, desc, now),
                )
                count += 1

        return count
