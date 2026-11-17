"""Tests for OpenAI-compatible error payloads in yourllm.core.exceptions."""

from yourllm.core.exceptions import (
    AuthenticationError,
    BadGatewayError,
    GatewayTimeoutError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    PayloadTooLargeError,
    PermissionError,
    RateLimitError,
    ServiceUnavailableError,
)


def test_authentication_error() -> None:
    err = AuthenticationError()
    assert err.status_code == 401
    assert err.error_type == "authentication_error"
    d = err.to_dict()
    assert d["error"]["type"] == "authentication_error"
    assert d["error"]["code"] == "invalid_api_key"
    assert "hint" in d["error"]


def test_permission_error() -> None:
    err = PermissionError("Forbidden")
    assert err.status_code == 403
    assert err.error_type == "permission_error"


def test_not_found_error() -> None:
    err = NotFoundError()
    assert err.status_code == 404
    assert err.error_type == "not_found_error"


def test_rate_limit_error() -> None:
    err = RateLimitError()
    assert err.status_code == 429
    assert err.error_type == "rate_limit_error"
    assert err.retry_after == 5


def test_service_unavailable_error() -> None:
    err = ServiceUnavailableError()
    assert err.status_code == 503
    assert err.error_type == "service_unavailable_error"
    assert err.retry_after == 10


def test_bad_gateway_and_timeout() -> None:
    bg = BadGatewayError()
    assert bg.status_code == 502
    gt = GatewayTimeoutError()
    assert gt.status_code == 504
    ie = InternalServerError()
    assert ie.status_code == 500
    pt = PayloadTooLargeError()
    assert pt.status_code == 413
    ir = InvalidRequestError("Malformed JSON")
    assert ir.status_code == 400

