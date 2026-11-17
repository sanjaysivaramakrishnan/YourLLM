"""Operational metrics collector and SQLite persistence integration."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import UTC, datetime
from typing import Any

from yourllm.storage.database import Database
from yourllm.storage.repositories import UsageStatsRepository

logger = logging.getLogger(__name__)

# Regex pattern to quickly match "usage": {...} inside raw SSE chunk bytes
USAGE_PATTERN = re.compile(
    rb'"usage"\s*:\s*\{[^}]*"prompt_tokens"\s*:\s*(\d+)[^}]*"completion_tokens"\s*:\s*(\d+)',
    re.DOTALL,
)


def extract_usage_from_sse_chunk(chunk: bytes) -> tuple[int, int] | None:
    """Inspect SSE chunk bytes (read-only tap) for token usage without buffering the stream.

    Per CLAUDE.md §3.4:
    The gateway forwards upstream SSE bytes without re-serializing. It may tap the stream
    to extract usage for metrics when the client requests it (e.g. stream_options.include_usage).

    Returns:
        (prompt_tokens, completion_tokens) if usage field found, else None.
    """
    if b'"usage"' not in chunk:
        return None

    match = USAGE_PATTERN.search(chunk)
    if match:
        try:
            prompt_tokens = int(match.group(1))
            completion_tokens = int(match.group(2))
            return prompt_tokens, completion_tokens
        except (ValueError, IndexError):
            pass

    # Fallback to JSON line extraction if regex did not capture due to key ordering
    for line in chunk.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(b"data:") and not trimmed.startswith(b"data: [DONE]"):
            try:
                payload = json.loads(trimmed[5:].strip())
                usage = payload.get("usage")
                if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
                    return int(usage["prompt_tokens"]), int(usage["completion_tokens"])
            except Exception:
                continue

    return None


class MetricsCollector:
    """Thread-safe operational metrics collector with aggregate SQLite flushing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.active_requests = 0
        self.total_latency_seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.upstream_errors = 0
        self.client_disconnects = 0
        self.rate_limit_hits = 0

        # Unflushed counters for periodic SQLite persistence
        self._unflushed_success = 0
        self._unflushed_failed = 0
        self._unflushed_prompt_tokens = 0
        self._unflushed_completion_tokens = 0
        self._unflushed_latency_ms = 0.0

    def request_started(self) -> None:
        """Increment active requests and total request counter."""
        with self._lock:
            self.total_requests += 1
            self.active_requests += 1

    def request_completed(
        self,
        latency_s: float,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record completed request outcome and token statistics."""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.total_latency_seconds += latency_s
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            latency_ms = latency_s * 1000.0
            if success:
                self.successful_requests += 1
                self._unflushed_success += 1
            else:
                self.failed_requests += 1
                self._unflushed_failed += 1

            self._unflushed_prompt_tokens += prompt_tokens
            self._unflushed_completion_tokens += completion_tokens
            self._unflushed_latency_ms += latency_ms

    def record_upstream_error(self) -> None:
        """Record an upstream backend failure."""
        with self._lock:
            self.upstream_errors += 1

    def record_client_disconnect(self) -> None:
        """Record a premature client connection drop."""
        with self._lock:
            self.client_disconnects += 1

    def record_rate_limit(self) -> None:
        """Record a 429 concurrency limit event."""
        with self._lock:
            self.rate_limit_hits += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a read-only dictionary of current in-memory metrics."""
        with self._lock:
            avg_latency = (
                round(self.total_latency_seconds / self.total_requests, 4)
                if self.total_requests > 0
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "active_requests": self.active_requests,
                "avg_latency_s": avg_latency,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "upstream_errors": self.upstream_errors,
                "client_disconnects": self.client_disconnects,
                "rate_limit_hits": self.rate_limit_hits,
            }

    def flush_to_db(self, db: Database, model_name: str) -> None:
        """Flush accumulated usage metrics into the SQLite usage_stats repository.

        Per CLAUDE.md §4:
        Never persist prompts or completions. Metrics are counts, latencies, and token totals only.
        """
        with self._lock:
            if (
                self._unflushed_success == 0
                and self._unflushed_failed == 0
                and self._unflushed_prompt_tokens == 0
                and self._unflushed_completion_tokens == 0
            ):
                return

            succ = self._unflushed_success
            fail = self._unflushed_failed
            p_tok = self._unflushed_prompt_tokens
            c_tok = self._unflushed_completion_tokens
            lat_ms = self._unflushed_latency_ms

            # Reset unflushed accumulators
            self._unflushed_success = 0
            self._unflushed_failed = 0
            self._unflushed_prompt_tokens = 0
            self._unflushed_completion_tokens = 0
            self._unflushed_latency_ms = 0.0

        period_hour = datetime.now(UTC).strftime("%Y-%m-%d %H:00:00")
        try:
            with db.connect() as conn:
                repo = UsageStatsRepository(conn)
                # Record successful requests portion
                if succ > 0 or p_tok > 0 or c_tok > 0:
                    repo.record_request(
                        model=model_name,
                        period_start=period_hour,
                        success=True,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        latency_ms=lat_ms if fail == 0 else (lat_ms * (succ / (succ + fail))),
                    )
                # Record failed requests portion
                if fail > 0:
                    repo.record_request(
                        model=model_name,
                        period_start=period_hour,
                        success=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=0.0 if succ == 0 else (lat_ms * (fail / (succ + fail))),
                    )
        except Exception as exc:
            logger.error("Failed to flush usage stats to database: %s", exc)
