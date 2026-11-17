"""API tests for streaming SSE, byte passthrough, disconnects, and concurrency limits."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fake_vllm import FakeVLLMServer, StreamingASGITransport
from yourllm.auth.service import create_key
from yourllm.core.config import Settings
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.inference.client import InferenceClient
from yourllm.main import create_app
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner


@pytest.fixture
def streaming_fixture(tmp_path: Path) -> dict:
    db = Database(tmp_path / "test.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    rec, raw_key = create_key(db, name="stream-test-key")

    fake_server = FakeVLLMServer(model_name="meta-llama/Llama-3.1-8B-Instruct")
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1-8b",
        YLLM_DATA_DIR=tmp_path,
        YLLM_MAX_CONCURRENT_REQUESTS=2,  # Cap concurrency at 2 for testing 429 backpressure
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.READY)
    inference_client = InferenceClient(
        base_url="http://fake-vllm",
        transport=StreamingASGITransport(fake_server.app),
    )

    app = create_app(
        settings=settings,
        db=db,
        lifecycle=lifecycle,
        inference_client=inference_client,
    )
    transport = StreamingASGITransport(app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    return {
        "app": app,
        "client": client,
        "key": raw_key,
        "fake_server": fake_server,
        "settings": settings,
        "lifecycle": lifecycle,
    }


@pytest.mark.asyncio
async def test_streaming_sse_byte_equality(streaming_fixture: dict) -> None:
    client: httpx.AsyncClient = streaming_fixture["client"]
    key: str = streaming_fixture["key"]
    fake_server: FakeVLLMServer = streaming_fixture["fake_server"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Write a Python script."}],
        "stream": True,
    }

    # Fetch expected bytes directly from fake server to assert byte-for-byte passthrough
    fake_transport = StreamingASGITransport(app=fake_server.app)
    async with httpx.AsyncClient(transport=fake_transport, base_url="http://fake-vllm") as direct_client:
        direct_resp = await direct_client.post("/v1/chat/completions", json=payload)
        expected_bytes = direct_resp.content

    # Now request through the gateway
    client_received_bytes = bytearray()
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        async for chunk in stream_resp.aiter_bytes():
            client_received_bytes.extend(chunk)

    # Gateway forwards upstream SSE bytes without re-serializing.
    # Assert exact byte equality, including [DONE] marker
    assert bytes(client_received_bytes) == expected_bytes
    assert b"data: [DONE]\n\n" in client_received_bytes


@pytest.mark.asyncio
async def test_streaming_tool_call_byte_equality(streaming_fixture: dict) -> None:
    client: httpx.AsyncClient = streaming_fixture["client"]
    key: str = streaming_fixture["key"]
    fake_server: FakeVLLMServer = streaming_fixture["fake_server"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Call pytest tool."}],
        "tools": [{"type": "function", "function": {"name": "run_command"}}],
        "stream": True,
    }

    fake_transport = StreamingASGITransport(app=fake_server.app)
    async with httpx.AsyncClient(transport=fake_transport, base_url="http://fake-vllm") as direct_client:
        direct_resp = await direct_client.post(
            "/v1/chat/completions", json=payload, headers={"X-Test-Mode": "tool_call"}
        )
        expected_bytes = direct_resp.content

    client_received_bytes = bytearray()
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "tool_call"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        async for chunk in stream_resp.aiter_bytes():
            client_received_bytes.extend(chunk)

    assert bytes(client_received_bytes) == expected_bytes
    assert b'"tool_calls"' in client_received_bytes
    assert b"data: [DONE]\n\n" in client_received_bytes


@pytest.mark.asyncio
async def test_streaming_incremental_flush(streaming_fixture: dict) -> None:
    client: httpx.AsyncClient = streaming_fixture["client"]
    key: str = streaming_fixture["key"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Incremental test."}],
        "stream": True,
    }

    chunks_timestamps: list[float] = []
    t_start = time.monotonic()

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "slow"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        async for line in stream_resp.aiter_lines():
            if line:
                chunks_timestamps.append(time.monotonic() - t_start)

    # Multiple distinct line arrivals prove incremental flush
    assert len(chunks_timestamps) >= 4
    first_chunk_latency = chunks_timestamps[0]
    total_stream_latency = chunks_timestamps[-1]
    assert first_chunk_latency < (total_stream_latency * 0.7)


@pytest.mark.asyncio
async def test_concurrency_cap_and_slot_release(streaming_fixture: dict) -> None:
    client: httpx.AsyncClient = streaming_fixture["client"]
    key: str = streaming_fixture["key"]
    fake_server: FakeVLLMServer = streaming_fixture["fake_server"]

    # Concurrency limit is set to 2 in streaming_fixture
    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hold query"}],
        "stream": True,
    }

    # Start 2 concurrent requests holding slots
    conn1 = client.stream(
        "POST", "/v1/chat/completions", json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "hold"},
    )
    conn2 = client.stream(
        "POST", "/v1/chat/completions", json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "hold"},
    )

    resp1 = await conn1.__aenter__()
    resp2 = await conn2.__aenter__()

    # Read first chunk from both to ensure slots are held inside the generator
    chunk1 = await anext(resp1.aiter_bytes())
    chunk2 = await anext(resp2.aiter_bytes())
    assert len(chunk1) > 0 and len(chunk2) > 0

    # 3rd request must be rejected with 429 RateLimitError
    resp3 = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp3.status_code == 429
    assert resp3.headers.get("Retry-After") == "5"
    assert resp3.json()["error"]["code"] == "concurrency_limit_exceeded"

    # Unblock fake server and close connection 1 to release slot
    fake_server.hold_event.set()
    await conn1.__aexit__(None, None, None)

    # Now 4th request should succeed because a slot was freed
    resp4 = await client.post(
        "/v1/chat/completions",
        json={"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Quick"}], "stream": False},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp4.status_code == 200

    # Cleanup connection 2
    await conn2.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_cancels_upstream(streaming_fixture: dict) -> None:
    client: httpx.AsyncClient = streaming_fixture["client"]
    key: str = streaming_fixture["key"]
    fake_server: FakeVLLMServer = streaming_fixture["fake_server"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Disconnect test"}],
        "stream": True,
    }

    # Open stream and read only the first chunk, then exit context (disconnect)
    conn = client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "slow"},
    )
    resp = await conn.__aenter__()
    assert resp.status_code == 200

    # Read 1 chunk
    chunk = await anext(resp.aiter_bytes())
    assert len(chunk) > 0

    # Prematurely close client connection
    await conn.__aexit__(None, None, None)

    # Allow a brief moment for the ASGI cancellation to propagate
    await asyncio.sleep(0.05)

    # Client disconnect must abort upstream vLLM generation
    assert fake_server.was_aborted is True
