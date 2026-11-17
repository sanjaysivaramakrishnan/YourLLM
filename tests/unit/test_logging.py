"""Tests for structured logging and secret redaction."""

import io
import json
import logging

from yourllm.core.logging import (
    clear_registered_secrets,
    redact_data,
    redact_text,
    register_secret,
    setup_logging,
)


def test_redact_api_key_patterns() -> None:
    """Standard API key formats must be scrubbed automatically."""
    raw = "User accessed using key yllm_abc123def456_supersecret32characterslongtoken"
    cleaned = redact_text(raw)
    assert "yllm_abc123def456_supersecret32characterslongtoken" not in cleaned
    assert "yllm_[REDACTED]" in cleaned


def test_redact_bearer_patterns() -> None:
    """Bearer tokens must be scrubbed."""
    raw = "Header Authorization: Bearer someSecretToken123456789"
    cleaned = redact_text(raw)
    assert "someSecretToken123456789" not in cleaned
    assert "Bearer [REDACTED]" in cleaned


def test_redact_hf_token_patterns() -> None:
    """HF tokens must be scrubbed."""
    raw = "Connecting with hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    cleaned = redact_text(raw)
    assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456" not in cleaned
    assert "hf_[REDACTED]" in cleaned


def test_planted_secret_in_captured_logs() -> None:
    """Planted custom secrets must never appear in formatted log records."""
    clear_registered_secrets()
    planted_secret = "PLANTED_CUSTOM_SECRET_987654321"
    register_secret(planted_secret)

    log_stream = io.StringIO()
    logger = setup_logging(level="DEBUG", json_format=True)

    # Swap stream to our capture stream
    for h in logger.root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = log_stream

    logger.info(f"Connecting upstream with secret: {planted_secret}")
    logger.debug(
        "Request processed",
        extra={"api_key": "yllm_key1_secretval", "custom": planted_secret},
    )

    output = log_stream.getvalue()

    # Secret literal must not exist anywhere in captured log text
    assert planted_secret not in output
    assert "secretval" not in output
    assert "[REDACTED]" in output

    # Must be valid single-line JSON records
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    assert len(lines) >= 2
    for line in lines:
        parsed = json.loads(line)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed

    clear_registered_secrets()


def test_redact_nested_structures() -> None:
    """Verify recursive redaction over dicts, lists, and tuples."""
    clear_registered_secrets()
    register_secret("SECRET_VAL")

    data = {
        "headers": {"auth": "Bearer token123"},
        "keys": ["yllm_k1_s1", "SECRET_VAL"],
        "tuple_data": ("normal", "yllm_k2_s2"),
    }
    redacted = redact_data(data)

    assert redacted["headers"]["auth"] == "Bearer [REDACTED]"
    assert redacted["keys"] == ["yllm_[REDACTED]", "[REDACTED]"]
    assert redacted["tuple_data"] == ("normal", "yllm_[REDACTED]")

    clear_registered_secrets()

