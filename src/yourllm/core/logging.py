"""JSON structured logging with sensitive secret redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

# Regex patterns for common API keys and authorization tokens
API_KEY_PATTERN = re.compile(r"yllm_[a-zA-Z0-9_-]+")
BEARER_PATTERN = re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]+", re.IGNORECASE)
HF_TOKEN_PATTERN = re.compile(r"hf_[a-zA-Z0-9]+")

_known_secrets: set[str] = set()


def register_secret(secret: str | None) -> None:
    """Register a plaintext secret string to be redacted from all log output."""
    if secret and len(secret.strip()) > 3:
        _known_secrets.add(secret.strip())


def clear_registered_secrets() -> None:
    """Clear all registered secrets (useful in test teardown)."""
    _known_secrets.clear()


def redact_text(text: str) -> str:
    """Scrub known secrets and recognized secret patterns from text."""
    if not text:
        return text

    # Redact registered secret literals
    for secret in _known_secrets:
        if secret in text:
            text = text.replace(secret, "[REDACTED]")

    # Redact key patterns
    text = API_KEY_PATTERN.sub("yllm_[REDACTED]", text)
    text = HF_TOKEN_PATTERN.sub("hf_[REDACTED]", text)
    text = BEARER_PATTERN.sub(r"\1[REDACTED]", text)

    return text


def redact_data(obj: Any) -> Any:
    """Recursively redact strings, dictionaries, lists, and tuples."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_data(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact_data(item) for item in obj)
    return obj


class RedactionFilter(logging.Filter):
    """Logging filter that scrubs sensitive secrets from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_data(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_data(arg) for arg in record.args)

        # Also scrub any extra attributes attached to the record
        for key, value in list(record.__dict__.items()):
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message",
            }:
                setattr(record, key, redact_data(value))

        return True


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include request_id if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """Configure the root logger with the redaction filter and JSON formatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(RedactionFilter())

    if json_format:
        stream_handler.setFormatter(JSONFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
        )

    root_logger.addHandler(stream_handler)
    return logging.getLogger("yourllm")
