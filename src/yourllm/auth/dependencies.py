"""FastAPI dependency callables for authenticating API keys and admin tokens."""

from __future__ import annotations

from typing import Annotated

from fastapi import Header, Request

from yourllm.auth.keys import constant_time_compare, parse_bearer_token
from yourllm.auth.service import validate_key
from yourllm.core.config import Settings
from yourllm.core.exceptions import AuthenticationError, NotFoundError
from yourllm.storage.database import Database
from yourllm.storage.repositories import APIKeyRecord


async def require_api_key(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> APIKeyRecord:
    """Validate Bearer inference key from the Authorization header."""
    if not authorization:
        raise AuthenticationError(
            message="Missing Authorization header.",
            code="missing_api_key",
            hint="Include 'Authorization: Bearer yllm_<key_id>_<secret>' in your request headers.",
        )

    try:
        raw_key = parse_bearer_token(authorization)
    except ValueError as exc:
        raise AuthenticationError(message=str(exc)) from exc

    # Retrieve database instance from app state
    db: Database | None = getattr(request.app.state, "db", None)
    if not db:
        # Fallback or misconfiguration
        raise AuthenticationError(message="Database authentication service is unavailable.")

    return validate_key(db, raw_key)


async def require_admin(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Validate admin token for /admin/* endpoints. Returns 404 when admin_token is unset."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    admin_token_secret = settings.admin_token if settings else None

    # CLAUDE.md §4: If unset, admin endpoints return 404 and the CLI is the only management path
    if not admin_token_secret or not admin_token_secret.get_secret_value():
        raise NotFoundError(message="The requested endpoint was not found.")

    if not authorization:
        raise AuthenticationError(
            message="Admin authentication required.",
            code="admin_auth_required",
            hint="Include 'Authorization: Bearer <YLLM_ADMIN_TOKEN>' in your request headers.",
        )

    try:
        token = parse_bearer_token(authorization)
    except ValueError as exc:
        raise AuthenticationError(message=str(exc)) from exc

    expected = admin_token_secret.get_secret_value()
    if not constant_time_compare(token, expected):
        raise AuthenticationError(
            message="Invalid admin token provided.",
            code="invalid_admin_token",
        )

    return token
