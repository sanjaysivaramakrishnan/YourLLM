"""Unit tests for API key lifecycle, verification, and hash-only storage."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yourllm.auth.dependencies import require_admin, require_api_key
from yourllm.auth.keys import (
    constant_time_compare,
    generate_api_key,
    parse_api_key,
    parse_bearer_token,
)
from yourllm.auth.service import (
    create_key,
    revoke_key,
    rotate_key,
    validate_key,
)
from yourllm.core.config import Settings
from yourllm.core.exceptions import AuthenticationError, NotFoundError
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """Fixture providing an initialized SQLite database with migrations applied."""
    db_file = tmp_path / "test_auth.db"
    db = Database(db_file)
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()
    return db


def test_key_format_and_uniqueness() -> None:
    """Verify standard yllm_<key_id>_<secret> structure and unique generation."""
    k1 = generate_api_key()
    k2 = generate_api_key()

    assert k1.plaintext_key.startswith("yllm_")
    assert k2.plaintext_key.startswith("yllm_")
    assert k1.key_id != k2.key_id
    assert k1.plaintext_key != k2.plaintext_key
    assert k1.key_hash != k2.key_hash

    # Parse key ID from generated key
    assert parse_api_key(k1.plaintext_key) == k1.key_id
    assert parse_api_key(k2.plaintext_key) == k2.key_id


def test_hash_only_storage(temp_db: Database) -> None:
    """Inspect raw SQLite DB file bytes to guarantee plaintext is never persisted."""
    record, plaintext = create_key(temp_db, name="vault-test")
    assert record.is_active

    # Read the raw SQLite database file on disk as bytes
    db_bytes = temp_db.db_path.read_bytes()

    # The plaintext key must NOT be found anywhere in the DB bytes
    assert plaintext.encode("utf-8") not in db_bytes

    # The secret part alone must not be found in the DB bytes
    secret_part = plaintext.split("_", 2)[2]
    assert secret_part.encode("utf-8") not in db_bytes

    # Only the key_id and the key_hash should exist
    assert record.key_id.encode("utf-8") in db_bytes
    assert record.key_hash.encode("utf-8") in db_bytes


def test_key_validation_and_revocation(temp_db: Database) -> None:
    """Verify validation succeeds for active keys and fails for revoked keys."""
    record, plaintext = create_key(temp_db, name="dev-key")

    # Valid key succeeds
    valid_rec = validate_key(temp_db, plaintext)
    assert valid_rec.key_id == record.key_id
    assert valid_rec.is_active

    # Revoke key
    revoked = revoke_key(temp_db, record.key_id)
    assert revoked is True

    # Revoked key must raise AuthenticationError
    with pytest.raises(AuthenticationError) as exc:
        validate_key(temp_db, plaintext)
    assert "revoked" in str(exc.value).lower()
    assert exc.value.code == "key_revoked"


def test_key_rotation(temp_db: Database) -> None:
    """Rotating a key revokes the original key and creates an active replacement."""
    old_rec, old_plaintext = create_key(temp_db, name="ci-key")

    new_rec, new_plaintext = rotate_key(temp_db, old_rec.key_id)

    assert new_rec.key_id != old_rec.key_id
    assert new_plaintext != old_plaintext
    assert new_rec.name == "ci-key (rotated)"
    assert new_rec.is_active

    # Old key is now revoked
    with pytest.raises(AuthenticationError):
        validate_key(temp_db, old_plaintext)

    # New key is valid
    validated = validate_key(temp_db, new_plaintext)
    assert validated.key_id == new_rec.key_id


def test_malformed_header_and_keys(temp_db: Database) -> None:
    """Header and key format edge cases must be safely rejected."""
    # Invalid bearer headers
    with pytest.raises(ValueError, match="Missing Authorization header"):
        parse_bearer_token(None)

    with pytest.raises(ValueError, match="Invalid Authorization header format"):
        parse_bearer_token("Basic dXNlcjpwYXNz")

    with pytest.raises(ValueError, match="Empty token"):
        parse_bearer_token("Bearer   ")

    # Invalid key formats
    with pytest.raises(ValueError, match="Invalid key prefix"):
        parse_api_key("sk_test_123456789")

    with pytest.raises(ValueError, match="Malformed API key"):
        parse_api_key("yllm_short")

    # Non-existent key
    with pytest.raises(AuthenticationError):
        validate_key(temp_db, "yllm_nonexistent_secrettoken123456789012345678")


def test_constant_time_comparison() -> None:
    """Verify timing-safe string comparison."""
    assert constant_time_compare("exact_match", "exact_match") is True
    assert constant_time_compare("match_a", "match_b") is False
    assert constant_time_compare("short", "longer_string") is False


@pytest.mark.asyncio
async def test_fastapi_require_api_key_dependency(temp_db: Database) -> None:
    """FastAPI require_api_key dependency validation."""
    record, plaintext = create_key(temp_db, name="api-dep-key")

    # Mock FastAPI request
    req = MagicMock()
    req.app.state.db = temp_db

    # Missing header raises 401
    with pytest.raises(AuthenticationError):
        await require_api_key(req, authorization=None)

    # Valid header returns record
    auth_header = f"Bearer {plaintext}"
    res = await require_api_key(req, authorization=auth_header)
    assert res.key_id == record.key_id


@pytest.mark.asyncio
async def test_fastapi_require_admin_dependency() -> None:
    """FastAPI require_admin dependency validation: 404 if unset, 401 if wrong."""
    req = MagicMock()

    # 1. Unset admin token -> 404 NotFoundError
    req.app.state.settings = Settings(YLLM_MODEL="dummy", YLLM_ADMIN_TOKEN=None)
    with pytest.raises(NotFoundError):
        await require_admin(req, authorization="Bearer some_token")

    # 2. Configured admin token
    req.app.state.settings = Settings(YLLM_MODEL="dummy", YLLM_ADMIN_TOKEN="adm_token_secret_123")

    # Missing auth -> 401
    with pytest.raises(AuthenticationError):
        await require_admin(req, authorization=None)

    # Wrong token -> 401
    with pytest.raises(AuthenticationError):
        await require_admin(req, authorization="Bearer wrong_token")

    # Correct token -> passes
    res = await require_admin(req, authorization="Bearer adm_token_secret_123")
    assert res == "adm_token_secret_123"
