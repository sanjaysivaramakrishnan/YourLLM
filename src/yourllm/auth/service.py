"""Authentication service managing keys, throttling, and verification."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from yourllm.auth.keys import (
    constant_time_compare,
    generate_api_key,
    hash_api_key,
    parse_api_key,
)
from yourllm.core.exceptions import AuthenticationError
from yourllm.storage.database import Database
from yourllm.storage.repositories import APIKeyRecord, APIKeyRepository

# In-memory throttle cache: key_id -> monotonic epoch of last database write
_last_used_write_cache: dict[str, float] = {}
LAST_USED_THROTTLE_SECONDS = 60.0


def create_key(db: Database, name: str) -> tuple[APIKeyRecord, str]:
    """Generate and store a new API key. Returns the record and one-time plaintext key."""
    trimmed_name = name.strip()
    if not trimmed_name:
        raise ValueError("Key name cannot be empty")

    gen = generate_api_key()
    with db.connect() as conn:
        repo = APIKeyRepository(conn)
        record = repo.create(
            key_id=gen.key_id,
            name=trimmed_name,
            key_hash=gen.key_hash,
        )
    return record, gen.plaintext_key


def list_keys(db: Database) -> list[APIKeyRecord]:
    """Retrieve metadata for all keys."""
    with db.connect() as conn:
        repo = APIKeyRepository(conn)
        return repo.list_keys()


def revoke_key(db: Database, key_id: str) -> bool:
    """Revoke an active key by its key_id."""
    with db.connect() as conn:
        repo = APIKeyRepository(conn)
        return repo.revoke(key_id)


def rotate_key(db: Database, key_id: str) -> tuple[APIKeyRecord, str]:
    """Rotate an existing key: marks old key revoked and returns a new active key."""
    with db.connect() as conn:
        repo = APIKeyRepository(conn)
        existing = repo.get_by_id(key_id)
        if not existing:
            raise ValueError(f"Key with ID '{key_id}' not found")
        if not existing.is_active:
            raise ValueError(f"Cannot rotate key '{key_id}' because it is already revoked")

        # Revoke old key
        repo.revoke(key_id)

        # Generate and create new key
        new_name = f"{existing.name} (rotated)"
        gen = generate_api_key()
        new_record = repo.create(
            key_id=gen.key_id,
            name=new_name,
            key_hash=gen.key_hash,
        )

    return new_record, gen.plaintext_key


def validate_key(db: Database, raw_key: str, throttle_last_used: bool = True) -> APIKeyRecord:
    """Verify an API key string, check active status, and record usage in constant time."""
    try:
        key_id = parse_api_key(raw_key)
    except ValueError as exc:
        raise AuthenticationError(message=str(exc)) from exc

    with db.connect() as conn:
        repo = APIKeyRepository(conn)
        record = repo.get_by_id(key_id)

    if not record:
        raise AuthenticationError()

    if not record.is_active:
        raise AuthenticationError(
            message="API key has been revoked.",
            code="key_revoked",
            hint="Generate a new key using the YourLLM CLI.",
        )

    # Compute hash of presented key and verify in constant time
    computed_hash = hash_api_key(raw_key)
    if not constant_time_compare(record.key_hash, computed_hash):
        raise AuthenticationError()

    # Throttled update of last_used_at
    if throttle_last_used:
        now_mono = time.monotonic()
        last_written = _last_used_write_cache.get(key_id, 0.0)
        if now_mono - last_written >= LAST_USED_THROTTLE_SECONDS:
            _last_used_write_cache[key_id] = now_mono
            now_iso = datetime.now(UTC).isoformat()
            try:
                with db.connect() as conn:
                    repo = APIKeyRepository(conn)
                    repo.update_last_used(key_id, now_iso)
            except Exception:
                # Do not block authorization if recording timestamp encounters an issue
                pass

    return record


def ensure_bootstrap_key(db: Database) -> str | None:
    """Create a bootstrap key only if no key has ever existed in the database.

    Returns the plaintext key string exactly once upon initial creation, or None.
    """
    with db.connect() as conn:
        repo = APIKeyRepository(conn)
        if repo.count() > 0:
            return None

    # Database has zero keys ever created: generate the bootstrap key
    _, plaintext_key = create_key(db, name="default-bootstrap")
    return plaintext_key
