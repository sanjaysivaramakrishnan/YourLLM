"""API tests for non-streaming chat completions, tool-calling, and error translations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tests.fixtures.fake_vllm import FakeVLLMServer
from yourllm.auth.service import create_key
from yourllm.core.config import Settings
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.inference.client import InferenceClient
from yourllm.main import create_app
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner


@pytest.fixture
def chat_fixture(tmp_path: Path) -> dict:
    db = Database(tmp_path / "test.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    rec, raw_key = create_key(db, name="chat-test-key")

    fake_server = FakeVLLMServer(model_name="meta-llama/Llama-3.1-8B-Instruct")
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1-8b",
        YLLM_DATA_DIR=tmp_path,
        YLLM_MAX_REQUEST_BYTES=2048,  # 2 KB for testing payload limits
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.READY)
    inference_client = InferenceClient(
        base_url="http://fake-vllm",
        transport=httpx.ASGITransport(app=fake_server.app),
    )

    app = create_app(
        settings=settings,
        db=db,
        lifecycle=lifecycle,
        inference_client=inference_client,
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    return {
        "client": client,
        "key": raw_key,
        "fake_server": fake_server,
        "lifecycle": lifecycle,
        "settings": settings,
        "inference_client": inference_client,
    }


@pytest.mark.asyncio
async def test_non_streaming_chat_completion_success(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Hello world!"},
        ],
        "temperature": 0.7,
        "stream": False,
    }

    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert data["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello from deterministic fake vLLM!"
    assert data["usage"]["total_tokens"] == 20
    assert resp.headers.get("X-Request-ID") is not None


@pytest.mark.asyncio
async def test_non_streaming_tool_call_response(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Please edit file src/app.py"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        "stream": False,
    }

    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "tool_call"},
    )
    assert resp.status_code == 200
    data = resp.json()

    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "edit_file"
    assert "src/app.py" in tool_calls[0]["function"]["arguments"]


@pytest.mark.asyncio
async def test_unknown_model_rejected(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_empty_messages_rejected(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    payload = {"model": "llama-3.1-8b", "messages": []}
    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 400
    assert "messages" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_malformed_json_rejected(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    resp = await client.post(
        "/v1/chat/completions",
        content=b"{invalid_json",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Malformed JSON" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_unsupported_modalities_rejected(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hello"}],
        "modalities": ["audio"],
    }
    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 400
    assert "Unsupported modalities" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_payload_too_large(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    # Limit is 2048 bytes; create 3000 bytes payload
    large_payload = {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "x" * 3000}],
    }
    resp = await client.post(
        "/v1/chat/completions",
        json=large_payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


@pytest.mark.asyncio
async def test_not_ready_state_returns_503(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]
    lifecycle: LifecycleManager = chat_fixture["lifecycle"]

    lifecycle.transition_to(LifecycleState.STOPPING)

    payload = {"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Hi"}]}
    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"
    assert resp.json()["error"]["code"] == "server_not_ready"


@pytest.mark.asyncio
async def test_upstream_500_maps_to_502(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]

    payload = {"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Hi"}]}
    resp = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {key}", "X-Test-Mode": "error_500"},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_server_error"


@pytest.mark.asyncio
async def test_upstream_timeout_maps_to_504(chat_fixture: dict) -> None:
    client: httpx.AsyncClient = chat_fixture["client"]
    key: str = chat_fixture["key"]
    inf_client: InferenceClient = chat_fixture["inference_client"]

    with patch.object(
        inf_client._client,
        "post",
        side_effect=httpx.ReadTimeout("Timeout generating token"),
    ):
        payload = {"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Hi"}]}
        resp = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
        )
        assert resp.status_code == 504
        assert resp.json()["error"]["code"] == "gateway_timeout"
