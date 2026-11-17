"""Unit tests for OpenAI-compatible response normalization layer."""

from __future__ import annotations

import json

from yourllm.api.normalizer import (
    StreamingResponseNormalizer,
    normalize_non_streaming_response,
)


def test_normalize_non_streaming_text_response_unchanged() -> None:
    """Standard text responses with finish_reason 'stop' remain untouched."""
    payload = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world!"},
                "finish_reason": "stop",
            }
        ],
    }
    raw = json.dumps(payload).encode("utf-8")
    normalized = normalize_non_streaming_response(raw)
    data = json.loads(normalized)
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["content"] == "Hello world!"


def test_normalize_non_streaming_tool_call_finish_reason_and_ids() -> None:
    """Tool-call responses with finish_reason 'stop' are corrected to 'tool_calls'."""
    payload = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            }
                        }
                    ],
                },
                "finish_reason": "stop",  # Defect from upstream vLLM
            }
        ],
    }
    raw = json.dumps(payload).encode("utf-8")
    normalized = normalize_non_streaming_response(raw)
    data = json.loads(normalized)

    choice = data["choices"][0]
    # 1. finish_reason must be "tool_calls"
    assert choice["finish_reason"] == "tool_calls"

    # 2. tool_call must have id and type
    tc = choice["message"]["tool_calls"][0]
    assert tc["id"].startswith("call_")
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "add"


def test_streaming_normalizer_text_flow() -> None:
    """Standard text streaming correctly yields chunks and ends with [DONE]."""
    normalizer = StreamingResponseNormalizer()

    chunk1 = b'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
    chunk2 = b'data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    chunk3 = b"data: [DONE]\n\n"

    events = []
    events.extend(normalizer.process_chunk(chunk1))
    events.extend(normalizer.process_chunk(chunk2))
    events.extend(normalizer.process_chunk(chunk3))
    events.extend(normalizer.finalize())

    assert len(events) == 3
    assert b"Hi" in events[0]
    d2 = json.loads(events[1].decode("utf-8").replace("data: ", "").strip())
    assert d2["choices"][0]["finish_reason"] == "stop"
    assert events[2] == b"data: [DONE]\n\n"


def test_streaming_normalizer_tool_call_delta_metadata_and_finish_reason() -> None:
    """Streaming tool call emits id and type on first delta and ensures finish_reason 'tool_calls'."""
    normalizer = StreamingResponseNormalizer()

    # Chunk 1: Delta starts a tool call, but upstream vLLM omitted 'id' and 'type'
    chunk1 = b'data: {"id":"chat-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"add","arguments":"{\\"a\\": 2"}}]},"finish_reason":null}]}\n\n'
    # Chunk 2: Upstream emits finish_reason: 'stop' (defect)
    chunk2 = b'data: {"id":"chat-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    # Chunk 3: Terminal marker
    chunk3 = b"data: [DONE]\n\n"

    events = []
    events.extend(normalizer.process_chunk(chunk1))
    events.extend(normalizer.process_chunk(chunk2))
    events.extend(normalizer.process_chunk(chunk3))
    events.extend(normalizer.finalize())

    # Verify Chunk 1 normalized: id and type injected
    d1 = json.loads(events[0].decode("utf-8").replace("data: ", "").strip())
    tc1 = d1["choices"][0]["delta"]["tool_calls"][0]
    assert tc1["id"].startswith("call_")
    assert tc1["type"] == "function"
    assert tc1["function"]["name"] == "add"

    # Verify Chunk 2 normalized: finish_reason changed from 'stop' to 'tool_calls'
    d2 = json.loads(events[1].decode("utf-8").replace("data: ", "").strip())
    assert d2["choices"][0]["finish_reason"] == "tool_calls"

    # Verify Chunk 3: [DONE]
    assert events[2] == b"data: [DONE]\n\n"


def test_streaming_normalizer_injects_finish_reason_if_missing_before_done() -> None:
    """If upstream abruptly sends [DONE] without any finish_reason, inject terminal finish chunk."""
    normalizer = StreamingResponseNormalizer()

    chunk1 = b'data: {"id":"chat-x","choices":[{"index":0,"delta":{"content":"Processing..."},"finish_reason":null}]}\n\n'
    chunk2 = b"data: [DONE]\n\n"  # Abrupt end without finish_reason chunk!

    events = []
    events.extend(normalizer.process_chunk(chunk1))
    events.extend(normalizer.process_chunk(chunk2))
    events.extend(normalizer.finalize())

    # Must contain 3 events: chunk1, injected terminal finish chunk ('stop'), and [DONE]
    assert len(events) == 3
    terminal = json.loads(events[1].decode("utf-8").replace("data: ", "").strip())
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert events[2] == b"data: [DONE]\n\n"
