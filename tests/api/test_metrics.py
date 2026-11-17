"""Tests for operational metrics, SSE usage extraction, and database persistence."""

from __future__ import annotations

from pathlib import Path

from yourllm.monitoring.metrics import MetricsCollector, extract_usage_from_sse_chunk
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner
from yourllm.storage.repositories import UsageStatsRepository


def test_extract_usage_from_sse_chunk() -> None:
    chunk_with_usage = (
        b'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":40,"total_tokens":65}}\n\n'
    )
    extracted = extract_usage_from_sse_chunk(chunk_with_usage)
    assert extracted == (25, 40)

    chunk_without_usage = (
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"}}]}\n\n'
    )
    assert extract_usage_from_sse_chunk(chunk_without_usage) is None

    done_chunk = b"data: [DONE]\n\n"
    assert extract_usage_from_sse_chunk(done_chunk) is None


def test_metrics_collector_and_sqlite_flush(tmp_path: Path) -> None:
    db = Database(tmp_path / "metrics_test.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    collector = MetricsCollector()
    collector.request_started()
    assert collector.active_requests == 1
    assert collector.total_requests == 1

    collector.request_completed(
        latency_s=0.25,
        success=True,
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert collector.active_requests == 0
    assert collector.successful_requests == 1
    assert collector.prompt_tokens == 100
    assert collector.completion_tokens == 50

    collector.record_upstream_error()
    assert collector.upstream_errors == 1

    collector.record_rate_limit()
    assert collector.rate_limit_hits == 1

    collector.record_client_disconnect()
    assert collector.client_disconnects == 1

    snap = collector.snapshot()
    assert snap["total_requests"] == 1
    assert snap["successful_requests"] == 1
    assert snap["avg_latency_s"] == 0.25

    # Flush to SQLite
    collector.flush_to_db(db, model_name="test-model")

    with db.connect() as conn:
        repo = UsageStatsRepository(conn)
        summary = repo.get_summary(model="test-model")
        assert len(summary) == 1
        row = summary[0]
        assert row["total_requests"] == 1
        assert row["successful_requests"] == 1
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 50

    # Ensure raw database content never stores prompt or completion text (privacy by design)
    with open(tmp_path / "metrics_test.db", "rb") as f:
        db_bytes = f.read()
        assert b"prompt_text" not in db_bytes
        assert b"completion_text" not in db_bytes

