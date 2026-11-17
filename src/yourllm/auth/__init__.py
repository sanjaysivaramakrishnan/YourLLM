"""Authentication package root."""

from yourllm.auth.dependencies import require_admin, require_api_key
from yourllm.auth.keys import (
    constant_time_compare,
    generate_api_key,
    hash_api_key,
    parse_api_key,
    parse_bearer_token,
)
from yourllm.auth.service import (
    create_key,
    ensure_bootstrap_key,
    list_keys,
    revoke_key,
    rotate_key,
    validate_key,
)

__all__ = [
    "constant_time_compare",
    "generate_api_key",
    "hash_api_key",
    "parse_api_key",
    "parse_bearer_token",
    "create_key",
    "ensure_bootstrap_key",
    "list_keys",
    "revoke_key",
    "rotate_key",
    "validate_key",
    "require_admin",
    "require_api_key",
]
