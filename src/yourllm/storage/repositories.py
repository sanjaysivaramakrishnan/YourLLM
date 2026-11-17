"""Repository classes encapsulating database queries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class APIKeyRecord:
    """Represents a persisted API key record (never contains plaintext)."""

    key_id: str
    name: str
    key_hash: str
    created_at: str
    last_used_at: str | None
    status: str
    revoked_at: str | None

    @property
    def is_active(self) -> bool:
        return self.status == "active"


class APIKeyRepository:
    """Data access layer for API keys."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, key_id: str, name: str, key_hash: str) -> APIKeyRecord:
        """Persist a newly generated API key hash and metadata."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """
            INSERT INTO api_keys (key_id, name, key_hash, created_at, last_used_at, status, revoked_at)
            VALUES (?, ?, ?, ?, NULL, 'active', NULL);
            """,
            (key_id, name, key_hash, now),
        )
        return APIKeyRecord(
            key_id=key_id,
            name=name,
            key_hash=key_hash,
            created_at=now,
            last_used_at=None,
            status="active",
            revoked_at=None,
        )

    def get_by_id(self, key_id: str) -> APIKeyRecord | None:
        """Look up a key record by its unique ID."""
        cur = self.conn.execute(
            """
            SELECT key_id, name, key_hash, created_at, last_used_at, status, revoked_at
            FROM api_keys
            WHERE key_id = ?;
            """,
            (key_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return APIKeyRecord(
            key_id=row["key_id"],
            name=row["name"],
            key_hash=row["key_hash"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            status=row["status"],
            revoked_at=row["revoked_at"],
        )

    def list_keys(self) -> list[APIKeyRecord]:
        """List all keys (ordered by creation date descending). Never reveals plaintext."""
        cur = self.conn.execute(
            """
            SELECT key_id, name, key_hash, created_at, last_used_at, status, revoked_at
            FROM api_keys
            ORDER BY created_at DESC;
            """
        )
        return [
            APIKeyRecord(
                key_id=row["key_id"],
                name=row["name"],
                key_hash=row["key_hash"],
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
                status=row["status"],
                revoked_at=row["revoked_at"],
            )
            for row in cur.fetchall()
        ]

    def revoke(self, key_id: str) -> bool:
        """Mark a key as revoked."""
        now = datetime.now(UTC).isoformat()
        cur = self.conn.execute(
            """
            UPDATE api_keys
            SET status = 'revoked', revoked_at = ?
            WHERE key_id = ? AND status = 'active';
            """,
            (now, key_id),
        )
        return cur.rowcount > 0

    def update_last_used(self, key_id: str, timestamp: str | None = None) -> None:
        """Update last_used_at timestamp for an active key."""
        ts = timestamp or datetime.now(UTC).isoformat()
        self.conn.execute(
            """
            UPDATE api_keys
            SET last_used_at = ?
            WHERE key_id = ?;
            """,
            (ts, key_id),
        )

    def count(self) -> int:
        """Count total keys ever created."""
        cur = self.conn.execute("SELECT COUNT(*) FROM api_keys;")
        row = cur.fetchone()
        return int(row[0]) if row else 0


class UsageStatsRepository:
    """Data access layer for aggregated usage metrics."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def record_request(
        self,
        model: str,
        period_start: str,
        success: bool,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record or aggregate metrics for a given model and time period."""
        succ_inc = 1 if success else 0
        fail_inc = 0 if success else 1
        self.conn.execute(
            """
            INSERT INTO usage_stats (
                model, period_start, total_requests, successful_requests, failed_requests,
                prompt_tokens, completion_tokens, total_latency_ms
            )
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(model, period_start) DO UPDATE SET
                total_requests = total_requests + 1,
                successful_requests = successful_requests + excluded.successful_requests,
                failed_requests = failed_requests + excluded.failed_requests,
                prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens,
                total_latency_ms = total_latency_ms + excluded.total_latency_ms;
            """,
            (
                model,
                period_start,
                succ_inc,
                fail_inc,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            ),
        )

    def get_summary(self, model: str | None = None) -> list[dict[str, Any]]:
        """Retrieve aggregated usage statistics."""
        if model:
            cur = self.conn.execute(
                """
                SELECT model,
                       SUM(total_requests) AS total_requests,
                       SUM(successful_requests) AS successful_requests,
                       SUM(failed_requests) AS failed_requests,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(total_latency_ms) AS total_latency_ms
                FROM usage_stats
                WHERE model = ?
                GROUP BY model;
                """,
                (model,),
            )
        else:
            cur = self.conn.execute(
                """
                SELECT model,
                       SUM(total_requests) AS total_requests,
                       SUM(successful_requests) AS successful_requests,
                       SUM(failed_requests) AS failed_requests,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(total_latency_ms) AS total_latency_ms
                FROM usage_stats
                GROUP BY model;
                """
            )
        return [dict(row) for row in cur.fetchall()]
