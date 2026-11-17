"""HTTP client wrapper communicating with the upstream internal inference engine."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from yourllm.core.exceptions import (
    BadGatewayError,
    GatewayTimeoutError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


class InferenceClient:
    """Async HTTP client communicating with the internal vLLM loopback server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        request_timeout_s: float = 600.0,
        connect_timeout_s: float = 5.0,
        max_connections: int = 100,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.connect_timeout_s = connect_timeout_s

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            timeouts = httpx.Timeout(
                timeout=request_timeout_s,
                connect=connect_timeout_s,
                read=request_timeout_s,
                write=15.0,
            )
            limits = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=20,
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeouts,
                limits=limits,
                transport=transport,
            )
            self._owns_client = True

    async def aclose(self) -> None:
        """Close client sessions and connection pools."""
        if self._owns_client:
            await self._client.aclose()

    async def check_health(self) -> bool:
        """Check upstream /health endpoint."""
        try:
            resp = await self._client.get("/health", timeout=self.connect_timeout_s)
            return resp.status_code == 200
        except Exception:
            return False

    async def get_models(self) -> httpx.Response:
        """Fetch models list from upstream /v1/models."""
        try:
            resp = await self._client.get("/v1/models")
            if resp.status_code >= 500:
                raise BadGatewayError(
                    f"Upstream inference engine returned status {resp.status_code}.",
                    code="upstream_server_error",
                )
            return resp
        except httpx.ConnectTimeout as exc:
            raise GatewayTimeoutError(
                "Connection to upstream inference engine timed out.",
                code="upstream_connect_timeout",
            ) from exc
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(
                "Upstream inference engine is currently unreachable.",
                code="upstream_unavailable",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise GatewayTimeoutError(
                "Upstream inference engine timed out reading models list.",
                code="gateway_timeout",
            ) from exc
        except (BadGatewayError, ServiceUnavailableError, GatewayTimeoutError):
            raise
        except Exception as exc:
            raise BadGatewayError(
                f"Failed to communicate with upstream inference engine: {exc}",
                code="upstream_communication_error",
            ) from exc

    async def post_chat_completions(
        self,
        payload: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        """Post non-streaming chat completions request to upstream."""
        # Sanitize headers forwarded upstream: drop host, content-length, authorization
        forward_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() in {"content-type", "accept", "x-request-id", "x-test-mode"}
        }
        if "content-type" not in {k.lower() for k in forward_headers}:
            forward_headers["content-type"] = "application/json"

        try:
            resp = await self._client.post(
                "/v1/chat/completions",
                content=payload,
                headers=forward_headers,
            )
            if resp.status_code >= 500:
                raise BadGatewayError(
                    f"Upstream inference engine returned status {resp.status_code}.",
                    code="upstream_server_error",
                )
            return resp
        except httpx.ConnectTimeout as exc:
            raise GatewayTimeoutError(
                "Connection to upstream inference engine timed out.",
                code="upstream_connect_timeout",
            ) from exc
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(
                "Upstream inference engine is currently unreachable.",
                code="upstream_unavailable",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise GatewayTimeoutError(
                "Upstream inference engine timed out generating chat completion.",
                code="gateway_timeout",
            ) from exc
        except (BadGatewayError, ServiceUnavailableError, GatewayTimeoutError):
            raise
        except Exception as exc:
            raise BadGatewayError(
                f"Failed to communicate with upstream inference engine: {exc}",
                code="upstream_communication_error",
            ) from exc

    @asynccontextmanager
    async def stream_chat_completions(
        self,
        payload: bytes,
        headers: dict[str, str],
    ) -> AsyncIterator[httpx.Response]:
        """Stream chat completions from upstream within an async context manager.

        Ensures the underlying upstream HTTP stream is explicitly closed if the caller
        exits, disconnects, or encounters an exception.
        """
        forward_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() in {"content-type", "accept", "x-request-id", "x-test-mode"}
        }
        if "content-type" not in {k.lower() for k in forward_headers}:
            forward_headers["content-type"] = "application/json"

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                content=payload,
                headers=forward_headers,
            ) as response:
                if response.status_code >= 500:
                    body = await response.aread()
                    raise BadGatewayError(
                        f"Upstream inference engine returned status {response.status_code}: {body.decode(errors='replace')[:200]}",
                        code="upstream_server_error",
                    )
                yield response
        except httpx.ConnectTimeout as exc:
            raise GatewayTimeoutError(
                "Connection to upstream inference engine timed out.",
                code="upstream_connect_timeout",
            ) from exc
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(
                "Upstream inference engine is currently unreachable.",
                code="upstream_unavailable",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise GatewayTimeoutError(
                "Upstream inference engine timed out during generation stream.",
                code="gateway_timeout",
            ) from exc
        except (BadGatewayError, ServiceUnavailableError, GatewayTimeoutError):
            raise
        except Exception as exc:
            raise BadGatewayError(
                f"Failed to stream from upstream inference engine: {exc}",
                code="upstream_stream_error",
            ) from exc
