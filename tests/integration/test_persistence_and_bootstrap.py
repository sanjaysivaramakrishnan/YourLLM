"""Integration tests for SQLite persistence, reopening, and bootstrap key generation."""

from pathlib import Path

from yourllm.auth.service import (
    create_key,
    ensure_bootstrap_key,
    validate_key,
)
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner
from yourllm.storage.repositories import UsageStatsRepository


def test_bootstrap_key_creation_once(tmp_path: Path) -> None:
    """Bootstrap key is generated exactly once on first boot and never re-created."""
    db_file = tmp_path / "bootstrap_test.db"
    db = Database(db_file)
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    # First call: No keys exist -> returns plaintext key
    first_bootstrap = ensure_bootstrap_key(db)
    assert first_bootstrap is not None
    assert first_bootstrap.startswith("yllm_")

    # Key is immediately functional
    record = validate_key(db, first_bootstrap)
    assert record.name == "default-bootstrap"
    assert record.is_active

    # Second call against the same DB: Keys exist -> returns None
    second_bootstrap = ensure_bootstrap_key(db)
    assert second_bootstrap is None

    # Even after creating additional keys or revoking, bootstrap key is never re-generated
    create_key(db, name="custom-app")
    third_bootstrap = ensure_bootstrap_key(db)
    assert third_bootstrap is None


def test_persistence_survives_reopening(tmp_path: Path) -> None:
    """Keys and stats persist on disk across new Database instances and connections."""
    db_file = tmp_path / "persistent_storage.db"

    # Step 1: Open initial instance and write records
    db1 = Database(db_file)
    with db1.connect() as conn:
        MigrationRunner(conn).apply_pending()

    rec1, key1 = create_key(db1, name="key-instance-1")

    with db1.connect() as conn:
        stats_repo = UsageStatsRepository(conn)
        stats_repo.record_request(
            model="test-model",
            period_start="2026-11-13T00:00:00Z",
            success=True,
            prompt_tokens=150,
            completion_tokens=50,
            latency_ms=120.5,
        )

    # Step 2: Simulate process termination and re-open fresh connection
    db2 = Database(db_file)
    with db2.connect() as conn:
        # Re-running migrations is idempotent
        applied = MigrationRunner(conn).apply_pending()
        assert applied == 0  # No new migrations

    # Verify key persists and validates cleanly
    validated = validate_key(db2, key1)
    assert validated.key_id == rec1.key_id
    assert validated.name == "key-instance-1"

    # Verify usage metrics persist
    with db2.connect() as conn:
        stats_repo = UsageStatsRepository(conn)
        summary = stats_repo.get_summary("test-model")
        assert len(summary) == 1
        assert summary[0]["total_requests"] == 1
        assert summary[0]["successful_requests"] == 1
        assert summary[0]["prompt_tokens"] == 150
        assert summary[0]["completion_tokens"] == 50
