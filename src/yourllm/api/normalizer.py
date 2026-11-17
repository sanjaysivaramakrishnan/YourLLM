"""OpenAI-compatible response normalization layer for chat completions.

Ensures strict OpenAI specification compliance for coding agents (Cline, Roo Code, Continue):
1. Maps finish_reason to 'tool_calls' whenever tool_calls are present in choices.
2. Injects tool_call id and type metadata on the initial streaming delta chunk.
3. Guarantees emission of a valid terminal finish_reason chunk before 'data: [DONE]'.
4. Prevents premature stream drop errors in client SDKs.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def normalize_non_streaming_response(raw_content: bytes) -> bytes:
    """Normalize a non-streaming chat completion JSON response.

    Ensures that if choices contain non-empty tool_calls, finish_reason is
    correctly reported as 'tool_calls' rather than 'stop', and each tool_call
    contains a valid 'id' and 'type': 'function'.
    """
    try:
        data = json.loads(raw_content)
    except Exception:
        return raw_content

    if not isinstance(data, dict):
        return raw_content

    modified = False
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            if choice.get("finish_reason") in ("stop", None):
                choice["finish_reason"] = "tool_calls"
                modified = True
            for tc in tool_calls:
                if isinstance(tc, dict):
                    if not tc.get("id"):
                        tc["id"] = f"call_{secrets.token_hex(12)}"
                        modified = True
                    if not tc.get("type"):
                        tc["type"] = "function"
                        modified = True

    if modified:
        return json.dumps(data).encode("utf-8")
    return raw_content


@dataclass
class StreamNormalizationState:
    """Tracks state across incremental SSE chunks in a streaming response."""

    buffer: str = ""
    has_seen_tool_calls: bool = False
    tool_call_ids: dict[int, str] = field(default_factory=dict)
    last_finish_reason: str | None = None
    last_id: str | None = None
    last_created: int | None = None
    last_model: str | None = None
    done_emitted: bool = False


class StreamingResponseNormalizer:
    """Normalizes SSE streams from vLLM to satisfy strict OpenAI streaming specifications."""

    def __init__(self) -> None:
        self.state = StreamNormalizationState()

    def process_chunk(self, chunk_bytes: bytes) -> list[bytes]:
        """Ingest raw incoming bytes and yield zero or more normalized SSE events."""
        text = chunk_bytes.decode("utf-8", errors="replace")
        self.state.buffer += text

        events: list[bytes] = []
        while "\n\n" in self.state.buffer:
            event_raw, self.state.buffer = self.state.buffer.split("\n\n", 1)
            event_list = self._normalize_event(event_raw)
            events.extend(event_list)

        return events

    def _normalize_event(self, event_str: str) -> list[bytes]:
        """Parse and normalize a single SSE event block (ending with double newline)."""
        lines = event_str.strip().split("\n")
        data_lines: list[str] = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("data:"):
                payload = line_stripped[5:].strip()
                data_lines.append(payload)

        if not data_lines:
            # Pass through comments or empty keep-alive pings verbatim
            return [(event_str + "\n\n").encode("utf-8")]

        payload_content = "\n".join(data_lines)

        # Handle terminal [DONE] marker
        if payload_content == "[DONE]":
            self.state.done_emitted = True
            out_events: list[bytes] = []
            # If the stream is ending but we haven't emitted a chunk with a finish_reason,
            # emit an explicit terminal chunk before [DONE] to satisfy client SDKs (Cline).
            if self.state.last_finish_reason is None or (
                self.state.has_seen_tool_calls and self.state.last_finish_reason in ("stop", None)
            ):
                target_reason = "tool_calls" if self.state.has_seen_tool_calls else "stop"
                terminal_chunk = {
                    "id": self.state.last_id or f"chatcmpl-{secrets.token_hex(8)}",
                    "object": "chat.completion.chunk",
                    "created": self.state.last_created or int(time.time()),
                    "model": self.state.last_model or "default",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": target_reason,
                        }
                    ],
                }
                out_events.append(f"data: {json.dumps(terminal_chunk)}\n\n".encode())
                self.state.last_finish_reason = target_reason

            out_events.append(b"data: [DONE]\n\n")
            return out_events

        # Parse JSON event
        try:
            chunk_data = json.loads(payload_content)
        except Exception:
            return [(event_str + "\n\n").encode("utf-8")]

        if not isinstance(chunk_data, dict):
            return [(event_str + "\n\n").encode("utf-8")]

        # Track metadata
        if chunk_data.get("id"):
            self.state.last_id = chunk_data["id"]
        if chunk_data.get("created"):
            self.state.last_created = chunk_data["created"]
        if chunk_data.get("model"):
            self.state.last_model = chunk_data["model"]

        # Inspect and normalize choices
        modified = False
        choices = chunk_data.get("choices", [])
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    tool_calls = delta.get("tool_calls")
                    if tool_calls and isinstance(tool_calls, list):
                        self.state.has_seen_tool_calls = True
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            idx = tc.get("index", 0)
                            if idx not in self.state.tool_call_ids:
                                # First time seeing this tool call index in stream:
                                # Ensure it has an id and type
                                tc_id = tc.get("id") or f"call_{secrets.token_hex(12)}"
                                self.state.tool_call_ids[idx] = tc_id
                                if tc.get("id") != tc_id:
                                    tc["id"] = tc_id
                                    modified = True
                                if tc.get("type") != "function":
                                    tc["type"] = "function"
                                    modified = True

                fr = choice.get("finish_reason")
                if fr is not None:
                    self.state.last_finish_reason = fr
                    # If tool calls were seen but finish_reason was emitted as 'stop', correct it
                    if self.state.has_seen_tool_calls and fr in ("stop", None):
                        choice["finish_reason"] = "tool_calls"
                        self.state.last_finish_reason = "tool_calls"
                        modified = True

        if modified:
            return [f"data: {json.dumps(chunk_data)}\n\n".encode()]
        return [(event_str + "\n\n").encode("utf-8")]

    def finalize(self) -> list[bytes]:
        """Called when upstream stream ends. Flushes buffers and guarantees [DONE]."""
        events: list[bytes] = []
        if self.state.buffer.strip():
            leftovers = self._normalize_event(self.state.buffer)
            events.extend(leftovers)
            self.state.buffer = ""

        # If [DONE] was never emitted (e.g. upstream dropped without sending [DONE]), emit it
        if not self.state.done_emitted:
            target_reason = "tool_calls" if self.state.has_seen_tool_calls else "stop"
            if self.state.last_finish_reason is None:
                terminal_chunk = {
                    "id": self.state.last_id or f"chatcmpl-{secrets.token_hex(8)}",
                    "object": "chat.completion.chunk",
                    "created": self.state.last_created or int(time.time()),
                    "model": self.state.last_model or "default",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": target_reason,
                        }
                    ],
                }
                events.append(f"data: {json.dumps(terminal_chunk)}\n\n".encode())
                self.state.last_finish_reason = target_reason
            events.append(b"data: [DONE]\n\n")
            self.state.done_emitted = True

        return events
