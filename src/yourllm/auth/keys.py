"""API key generation, hashing, and parsing primitives."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedKey:
    """Holds a newly generated key ID, hash, and one-time plaintext key."""

    key_id: str
    plaintext_key: str
    key_hash: str


def generate_key_id() -> str:
    """Generate a collision-resistant 10-character hex key identifier."""
    return secrets.token_hex(5)


def generate_api_key(key_id: str | None = None) -> GeneratedKey:
    """Generate a standard formatted API key: yllm_<key_id>_<secret>."""
    kid = key_id or generate_key_id()
    secret = secrets.token_urlsafe(32)
    plaintext = f"yllm_{kid}_{secret}"
    key_hash = hash_api_key(plaintext)
    return GeneratedKey(key_id=kid, plaintext_key=plaintext, key_hash=key_hash)


def hash_api_key(key: str) -> str:
    """Compute deterministic SHA-256 hash of a full API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def constant_time_compare(val_a: str, val_b: str) -> bool:
    """Compare two string values in constant time to prevent timing attacks."""
    return hmac.compare_digest(val_a.encode("utf-8"), val_b.encode("utf-8"))


def parse_bearer_token(header_value: str | None) -> str:
    """Extract token string from an 'Authorization: Bearer <token>' header value."""
    if not header_value:
        raise ValueError("Missing Authorization header")

    header_clean = header_value.strip()
    if not header_clean.lower().startswith("bearer"):
        raise ValueError("Invalid Authorization header format. Expected 'Bearer <token>'")

    token_part = header_clean[6:].strip()
    if not token_part:
        raise ValueError("Empty token in Authorization header")

    return token_part


def parse_api_key(key: str) -> str:
    """Parse key string and extract key_id. Must match format 'yllm_<key_id>_<secret>'."""
    if not key.startswith("yllm_"):
        raise ValueError("Invalid key prefix. API keys must start with 'yllm_'")

    parts = key.split("_")
    # Expected format: ['yllm', key_id, secret]
    if len(parts) < 3:
        raise ValueError("Malformed API key structure")

    key_id = parts[1]
    if not key_id:
        raise ValueError("Missing key ID in API key")

    return key_id
