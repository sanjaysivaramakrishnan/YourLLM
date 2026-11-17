"""OpenAI-compatible request and response schemas for chat and models endpoints."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ModelPermission(BaseModel):
    """OpenAI model permission sub-object."""

    id: str = "modelperm-default"
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelObject(BaseModel):
    """OpenAI Model description object."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "yourllm"
    root: str | None = None
    parent: str | None = None
    permission: list[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])


class ModelListResponse(BaseModel):
    """OpenAI list models response."""

    object: str = "list"
    data: list[ModelObject]


class ChatMessage(BaseModel):
    """Single message within chat completion history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant", "tool"}
        if v not in allowed:
            raise ValueError(f"Invalid message role '{v}'. Allowed roles: {sorted(allowed)}")
        return v


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool | None = False


class ChatCompletionRequest(BaseModel):
    """Request payload for /v1/chat/completions."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1, le=1)  # MVP supports single completion (n=1)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None

    # Unsupported modalities rejection per M3 specification
    modalities: list[str] | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("'messages' list must contain at least one message.")
        return v

    @field_validator("modalities")
    @classmethod
    def reject_unsupported_modalities(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            non_text = [m for m in v if m != "text"]
            if non_text:
                raise ValueError(
                    f"Unsupported modalities requested: {non_text}. Only 'text' is supported."
                )
        return v

