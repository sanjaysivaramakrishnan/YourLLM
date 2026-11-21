"""OpenAI-compatible chat completions route (/v1/chat/completions)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from yourllm.api.normalizer import StreamingResponseNormalizer, normalize_non_streaming_response
from yourllm.api.schemas import ChatCompletionRequest
from yourllm.auth.dependencies import require_api_key
from yourllm.core.config import Settings
from yourllm.core.exceptions import (
    InvalidRequestError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitError,
    ServiceUnavailableError,
)
from yourllm.core.lifecycle import LifecycleManager
from yourllm.inference.client import InferenceClient
from yourllm.monitoring.metrics import MetricsCollector, extract_usage_from_sse_chunk
from yourllm.storage.repositories import APIKeyRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ConcurrencyLimiter:
    """Async counter enforcing backpressure concurrency limits (CLAUDE.md §3.10)."""

    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max_concurrent
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return self._active

    async def try_acquire(self) -> bool:
        """Attempt to acquire an active concurrency slot. Returns False if full."""
        async with self._lock:
            if self._active >= self.max_concurrent:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        """Release an active concurrency slot."""
        async with self._lock:
            self._active = max(0, self._active - 1)


def _check_ready(request: Request) -> None:
    """Ensure system state is READY before serving inference."""
    lifecycle: LifecycleManager | None = getattr(request.app.state, "lifecycle", None)
    if lifecycle and not lifecycle.is_ready:
        raise ServiceUnavailableError(
            message="Model is loading or server is not ready.",
            code="server_not_ready",
            retry_after=10,
            hint="Poll /health/ready until status is READY.",
        )


async def _read_and_validate_body(request: Request, max_bytes: int) -> bytes:
    """Read request body while enforcing size limits (HTTP 413)."""
    # Check Content-Length header if present
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length_val = int(content_length)
            if length_val > max_bytes:
                raise PayloadTooLargeError(
                    f"Request payload ({length_val} bytes) exceeds maximum limit ({max_bytes} bytes)."
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > max_bytes:
        raise PayloadTooLargeError(
            f"Request payload ({len(body)} bytes) exceeds maximum limit ({max_bytes} bytes)."
        )
    return body


@router.post("/completions")
async def chat_completions(
    request: Request,
    _key_record: Annotated[APIKeyRecord, Depends(require_api_key)],
) -> Response:
    """Proxy OpenAI chat completions request with streaming and tool calling support."""
    # 1. State check: must be in READY state
    _check_ready(request)

    settings: Settings = request.app.state.settings
    inference_client: InferenceClient = request.app.state.inference_client
    metrics: MetricsCollector = request.app.state.metrics
    limiter: ConcurrencyLimiter = request.app.state.concurrency_limiter

    # 2. Enforce request size limit
    raw_body = await _read_and_validate_body(request, settings.max_request_bytes)

    # 3. Parse and validate JSON schema
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(f"Malformed JSON in request body: {exc}") from exc

    if not isinstance(data, dict):
        raise InvalidRequestError("Request body must be a JSON object.")

    try:
        parsed_req = ChatCompletionRequest(**data)
    except Exception as exc:
        raise InvalidRequestError(f"Request validation error: {exc}") from exc

    # 4. Model validation
    served_name = settings.served_model_name or settings.model
    if parsed_req.model not in {served_name, settings.model}:
        raise NotFoundError(
            message=f"The model '{parsed_req.model}' does not exist.",
            code="model_not_found",
            hint=f"Configured model identifier: '{served_name}'.",
        )

    # Normalize model in upstream payload to served_name so vLLM model check always matches
    if data.get("model") != served_name:
        data["model"] = served_name
        raw_body = json.dumps(data).encode("utf-8")

    # 5. Concurrency backpressure check
    acquired = await limiter.try_acquire()
    if not acquired:
        metrics.record_rate_limit()
        raise RateLimitError(
            message=f"Server concurrency limit reached ({settings.max_concurrent_requests} concurrent requests).",
            retry_after=5,
            hint="Please retry after a brief delay.",
        )

    metrics.request_started()
    start_time = time.monotonic()

    # 6. Streaming flow (SSE byte passthrough)
    if parsed_req.stream:
        return _handle_streaming(
            request=request,
            raw_body=raw_body,
            inference_client=inference_client,
            metrics=metrics,
            limiter=limiter,
            start_time=start_time,
        )

    # 7. Non-streaming flow
    return await _handle_non_streaming(
        request=request,
        raw_body=raw_body,
        inference_client=inference_client,
        metrics=metrics,
        limiter=limiter,
        start_time=start_time,
    )


async def _handle_non_streaming(
    request: Request,
    raw_body: bytes,
    inference_client: InferenceClient,
    metrics: MetricsCollector,
    limiter: ConcurrencyLimiter,
    start_time: float,
) -> Response:
    """Execute non-streaming completion request."""
    success = False
    p_tokens = 0
    c_tokens = 0

    try:
        upstream_resp = await inference_client.post_chat_completions(
            payload=raw_body,
            headers=dict(request.headers),
        )
        latency = time.monotonic() - start_time
        success = upstream_resp.status_code == 200

        if success:
            try:
                resp_json = upstream_resp.json()
                usage = resp_json.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
            except Exception:
                pass
        else:
            logger.warning(
                "Upstream vLLM returned HTTP %d for chat completion: %s",
                upstream_resp.status_code,
                upstream_resp.text[:500],
            )
            metrics.record_upstream_error()

        metrics.request_completed(
            latency_s=latency,
            success=success,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
        )

        # Normalize response JSON for strict OpenAI tool call compliance
        content = upstream_resp.content
        if success:
            content = normalize_non_streaming_response(content)

        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers={"Content-Type": "application/json"},
        )

    except Exception:
        metrics.record_upstream_error()
        metrics.request_completed(
            latency_s=time.monotonic() - start_time,
            success=False,
        )
        raise

    finally:
        await limiter.release()


def _handle_streaming(
    request: Request,
    raw_body: bytes,
    inference_client: InferenceClient,
    metrics: MetricsCollector,
    limiter: ConcurrencyLimiter,
    start_time: float,
) -> StreamingResponse:
    """Execute streaming completion request with OpenAI-compliant SSE normalization.

    Ensures proper tool-call metadata propagation, correct finish_reason ('tool_calls'),
    and clean terminal [DONE] emission to prevent coding agent stream errors (Cline).
    Client disconnect terminates upstream to abort generation (CLAUDE.md §3.5).
    """

    async def _stream_generator() -> AsyncIterator[bytes]:
        success = False
        prompt_tokens = 0
        completion_tokens = 0
        normalizer = StreamingResponseNormalizer()

        try:
            async with inference_client.stream_chat_completions(
                payload=raw_body,
                headers=dict(request.headers),
            ) as stream_resp:
                success = stream_resp.status_code == 200
                if not success:
                    err_bytes = await stream_resp.aread()
                    logger.warning(
                        "Upstream vLLM streaming returned HTTP %d: %s",
                        stream_resp.status_code,
                        err_bytes.decode(errors="replace")[:500],
                    )
                    yield err_bytes
                    return
                async for chunk in stream_resp.aiter_raw():
                    # Check for client disconnect
                    if await request.is_disconnected():
                        metrics.record_client_disconnect()
                        logger.info("Client disconnected mid-stream; cancelling upstream inference.")
                        break

                    # Read-only tap for token metrics
                    usage = extract_usage_from_sse_chunk(chunk)
                    if usage:
                        prompt_tokens, completion_tokens = usage

                    for event in normalizer.process_chunk(chunk):
                        yield event

            # End of upstream stream: flush any leftover buffer or guarantee terminal finish_reason chunk
            for event in normalizer.finalize():
                yield event

        except (asyncio.CancelledError, GeneratorExit):
            metrics.record_client_disconnect()
            logger.info("Client disconnected / cancelled mid-stream.")
            raise
        except Exception:
            metrics.record_upstream_error()
            raise

        finally:
            latency = time.monotonic() - start_time
            metrics.request_completed(
                latency_s=latency,
                success=success,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            await limiter.release()

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
