"""Deterministic test-only fake vLLM server ASGI application.

Strictly test-only. Excluded from the production package and Docker image.
Simulates internal vLLM /health, /v1/models, and /v1/chat/completions (streaming,
non-streaming, tool-calls, slow streams, and mid-stream disconnects).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route


class StreamingResponseStream(httpx.AsyncByteStream):
    """Asynchronous byte stream reading incrementally from a queue."""

    def __init__(
        self,
        queue: asyncio.Queue[bytes | None],
        on_close: Callable[[], None],
        transport: StreamingASGITransport,
    ) -> None:
        self._queue = queue
        self._on_close = on_close
        self._transport = transport

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if self._transport.app_exception:
                raise self._transport.app_exception[0]
            if chunk is None:
                break
            yield chunk

    async def aclose(self) -> None:
        self._on_close()


class StreamingASGITransport(httpx.AsyncBaseTransport):
    """Test transport providing true incremental streaming and disconnect support for ASGI apps."""

    def __init__(self, app: Any, raise_app_exceptions: bool = True) -> None:
        self.app = app
        self.raise_app_exceptions = raise_app_exceptions
        self.app_exception: list[BaseException] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
        }

        request_body_chunks = request.stream.__aiter__()
        request_complete = False
        client_disconnected = asyncio.Event()
        receive_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def receive() -> dict[str, Any]:
            nonlocal request_complete
            if not receive_queue.empty():
                return await receive_queue.get()
            if client_disconnected.is_set():
                return {"type": "http.disconnect"}
            if request_complete:
                await client_disconnected.wait()
                return {"type": "http.disconnect"}
            try:
                chunk = await request_body_chunks.__anext__()
                return {"type": "http.request", "body": chunk, "more_body": True}
            except StopAsyncIteration:
                request_complete = True
                return {"type": "http.request", "body": b"", "more_body": False}

        status_code = None
        response_headers = None
        response_started = asyncio.Event()
        body_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        app_task: asyncio.Task[None] | None = None

        async def send(message: dict[str, Any]) -> None:
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = message.get("headers", [])
                response_started.set()
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                more_body = message.get("more_body", False)
                if body and request.method != "HEAD":
                    await body_queue.put(body)
                if not more_body:
                    await body_queue.put(None)

        def on_close() -> None:
            client_disconnected.set()
            receive_queue.put_nowait({"type": "http.disconnect"})
            if app_task and not app_task.done():
                app_task.cancel()

        async def run_app() -> None:
            try:
                await self.app(scope, receive, send)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if self.raise_app_exceptions:
                    self.app_exception.append(exc)
            finally:
                await body_queue.put(None)
                response_started.set()

        app_task = asyncio.create_task(run_app())
        await response_started.wait()

        if self.app_exception:
            raise self.app_exception[0]

        if status_code is None:
            status_code = 500
            response_headers = []

        stream = StreamingResponseStream(body_queue, on_close=on_close, transport=self)
        return httpx.Response(status_code, headers=response_headers, stream=stream)


class FakeVLLMServer:
    """Stateful fake vLLM server application for testing the gateway."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        self.model_name = model_name
        self.requests_received: list[dict[str, Any]] = []
        self.was_aborted = False
        self.chunks_sent = 0
        self.hold_event = asyncio.Event()
        self.app = Starlette(
            routes=[
                Route("/health", self.health, methods=["GET"]),
                Route("/v1/models", self.models, methods=["GET"]),
                Route("/v1/chat/completions", self.chat_completions, methods=["POST"]),
            ]
        )

    @property
    def base_url(self) -> str:
        return "http://fake-vllm"

    @property
    def is_ready(self) -> bool:
        return True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def check_health(self) -> bool:
        return True

    def get_transport(self) -> StreamingASGITransport:
        """Return a StreamingASGITransport wired to this fake server's app."""
        return StreamingASGITransport(self.app)

    def reset(self) -> None:
        """Reset state tracking between tests."""
        self.requests_received.clear()
        self.was_aborted = False
        self.chunks_sent = 0
        self.hold_event.clear()

    async def health(self, request: Request) -> Response:
        return PlainTextResponse("OK", status_code=200)

    async def models(self, request: Request) -> Response:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "vllm",
                    }
                ],
            }
        )

    async def chat_completions(self, request: Request) -> Response:
        test_mode = request.headers.get("x-test-mode", "normal")

        if test_mode == "error_500":
            return JSONResponse(
                {"error": {"message": "Internal vLLM engine crash simulated.", "type": "server_error"}},
                status_code=500,
            )

        if test_mode == "malformed_json":
            return Response(content="not a valid json string", status_code=200, media_type="application/json")

        body = await request.body()
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        self.requests_received.append({"payload": payload, "headers": dict(request.headers)})
        is_stream = payload.get("stream", False)

        if test_mode == "tool_call":
            if is_stream:
                return self._stream_tool_call(request)
            return self._non_stream_tool_call()

        if is_stream:
            is_slow = test_mode == "slow"
            is_hold = test_mode == "hold"
            return self._stream_normal(request, is_slow=is_slow, is_hold=is_hold)

        return self._non_stream_normal()

    def _non_stream_normal(self) -> Response:
        return JSONResponse(
            {
                "id": "chatcmpl-fake-normal",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from deterministic fake vLLM!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            }
        )

    def _non_stream_tool_call(self) -> Response:
        return JSONResponse(
            {
                "id": "chatcmpl-fake-tool",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "edit_file",
                                        "arguments": '{"path": "src/app.py", "content": "print(1)"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 18,
                    "total_tokens": 43,
                },
            }
        )

    def _stream_normal(
        self,
        request: Request,
        is_slow: bool = False,
        is_hold: bool = False,
    ) -> StreamingResponse:
        chunks = [
            b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}\n\n',
            b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" from"}}]}\n\n',
            b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" streaming"}}]}\n\n',
            b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" fake vLLM!"}}]}\n\n',
            b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":10,"total_tokens":25}}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def generator() -> AsyncIterator[bytes]:
            try:
                for idx, chunk in enumerate(chunks):
                    if await request.is_disconnected():
                        self.was_aborted = True
                        break
                    if is_hold and idx == 0:
                        self.chunks_sent += 1
                        yield chunk
                        try:
                            await asyncio.wait_for(self.hold_event.wait(), timeout=10.0)
                        except TimeoutError:
                            pass
                        continue
                    if is_slow:
                        await asyncio.sleep(0.05)
                    self.chunks_sent += 1
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                self.was_aborted = True
                raise

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
        )

    def _stream_tool_call(self, request: Request) -> StreamingResponse:
        chunks = [
            b'data: {"id":"chatcmpl-st1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_999","type":"function","function":{"name":"run_command","arguments":""}}]}}]}\n\n',
            b'data: {"id":"chatcmpl-st1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"command\\": \\"pytest\\"}"}}]}}]}\n\n',
            b'data: {"id":"chatcmpl-st1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":30,"completion_tokens":12,"total_tokens":42}}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def generator() -> AsyncIterator[bytes]:
            try:
                for chunk in chunks:
                    if await request.is_disconnected():
                        self.was_aborted = True
                        break
                    self.chunks_sent += 1
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                self.was_aborted = True
                raise

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
        )
