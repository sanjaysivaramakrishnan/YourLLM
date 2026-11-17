"""Configuration management for YourLLM via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """YourLLM runtime settings parsed from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Model configuration
    model: str = Field(
        ...,
        alias="YLLM_MODEL",
        description="Hugging Face repo ID or local model path (required)",
    )
    served_model_name: str | None = Field(
        default=None,
        alias="YLLM_SERVED_MODEL_NAME",
        description="Model name exposed via the OpenAI-compatible API",
    )
    profile: str = Field(
        default="auto",
        alias="YLLM_PROFILE",
        description="Model profile name or 'auto' to match by HF repo ID",
    )
    hf_token: SecretStr | None = Field(
        default=None,
        alias="HF_TOKEN",
        description="Hugging Face access token for gated models",
    )

    # Gateway network and security
    port: int = Field(
        default=8000,
        alias="YLLM_PORT",
        description="Gateway listening port",
    )
    vllm_port: int = Field(
        default=8001,
        alias="YLLM_INTERNAL_VLLM_PORT",
        description="Internal loopback port for the supervised vLLM subprocess",
    )
    vllm_python: str | None = Field(
        default=None,
        alias="YLLM_VLLM_PYTHON",
        description="Path to Python interpreter for vLLM subprocess (defaults to /usr/bin/python3 in container)",
    )
    data_dir: Path = Field(
        default=Path("/data"),
        alias="YLLM_DATA_DIR",
        description="Base persistent data directory",
    )
    admin_token: SecretStr | None = Field(
        default=None,
        alias="YLLM_ADMIN_TOKEN",
        description="Secret token for accessing /admin/* HTTP endpoints",
    )
    max_concurrent_requests: int = Field(
        default=16,
        alias="YLLM_MAX_CONCURRENT_REQUESTS",
        description="Gateway concurrency limit",
    )
    request_timeout_s: float = Field(
        default=600.0,
        alias="YLLM_REQUEST_TIMEOUT_S",
        description="Per-request upstream timeout in seconds",
    )
    max_request_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        alias="YLLM_MAX_REQUEST_BYTES",
        description="Maximum request body size in bytes",
    )
    log_level: str = Field(
        default="INFO",
        alias="YLLM_LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # vLLM Engine tuning
    max_model_len: int | None = Field(
        default=None,
        alias="YLLM_MAX_MODEL_LEN",
        description="Context length ceiling in tokens",
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        alias="YLLM_GPU_MEMORY_UTILIZATION",
        description="Fraction of GPU VRAM vLLM may use (0.0, 1.0]",
    )
    dtype: str = Field(
        default="auto",
        alias="YLLM_DTYPE",
        description="Model weight data type precision",
    )
    quantization: str | None = Field(
        default=None,
        alias="YLLM_QUANTIZATION",
        description="Quantization method (awq, gptq, fp8, etc.)",
    )
    tensor_parallel_size: int = Field(
        default=1,
        alias="YLLM_TENSOR_PARALLEL_SIZE",
        description="Single-node multi-GPU tensor parallel count",
    )
    max_num_seqs: int | None = Field(
        default=None,
        alias="YLLM_MAX_NUM_SEQS",
        description="vLLM max concurrent sequences per forward pass",
    )
    enable_prefix_caching: bool = Field(
        default=True,
        alias="YLLM_ENABLE_PREFIX_CACHING",
        description="Enable automatic prefix caching",
    )
    kv_cache_dtype: str = Field(
        default="auto",
        alias="YLLM_KV_CACHE_DTYPE",
        description="KV cache precision (auto, fp8, fp8_e4m3, fp8_e5m2)",
    )
    tool_call_parser: str | None = Field(
        default=None,
        alias="YLLM_TOOL_CALL_PARSER",
        description="Tool call parser override",
    )
    chat_template: str | None = Field(
        default=None,
        alias="YLLM_CHAT_TEMPLATE",
        description="Custom chat template path override",
    )
    trust_remote_code: bool = Field(
        default=False,
        alias="YLLM_TRUST_REMOTE_CODE",
        description="Trust remote code from Hugging Face repository",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("YLLM_MODEL cannot be empty")
        return trimmed

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_memory_utilization(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"YLLM_GPU_MEMORY_UTILIZATION must be in range (0.0, 1.0], got {v}"
            )
        return v

    @field_validator("tensor_parallel_size")
    @classmethod
    def validate_tensor_parallel_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"YLLM_TENSOR_PARALLEL_SIZE must be >= 1, got {v}"
            )
        return v

    @field_validator("max_concurrent_requests")
    @classmethod
    def validate_max_concurrent_requests(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"YLLM_MAX_CONCURRENT_REQUESTS must be >= 1, got {v}"
            )
        return v

    @field_validator("request_timeout_s")
    @classmethod
    def validate_request_timeout_s(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"YLLM_REQUEST_TIMEOUT_S must be > 0, got {v}"
            )
        return v

    @field_validator("max_request_bytes")
    @classmethod
    def validate_max_request_bytes(cls, v: int) -> int:
        if v < 1024:
            raise ValueError(
                f"YLLM_MAX_REQUEST_BYTES must be at least 1024 bytes, got {v}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        upper = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if upper not in allowed:
            raise ValueError(
                f"YLLM_LOG_LEVEL must be one of {sorted(allowed)}, got {v}"
            )
        return upper

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        lower = v.strip().lower()
        allowed = {"auto", "half", "float16", "bfloat16", "float32"}
        if lower not in allowed:
            raise ValueError(
                f"YLLM_DTYPE must be one of {sorted(allowed)}, got {v}"
            )
        return lower

    @field_validator("kv_cache_dtype")
    @classmethod
    def validate_kv_cache_dtype(cls, v: str) -> str:
        lower = v.strip().lower()
        allowed = {"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}
        if lower not in allowed:
            raise ValueError(
                f"YLLM_KV_CACHE_DTYPE must be one of {sorted(allowed)}, got {v}"
            )
        return lower

    @field_validator("max_model_len")
    @classmethod
    def validate_max_model_len(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"YLLM_MAX_MODEL_LEN must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def resolve_served_model_name(self) -> Settings:
        if not self.served_model_name:
            self.served_model_name = self.model
        return self

    @property
    def database_path(self) -> Path:
        """Absolute path to the SQLite persistence database."""
        return self.data_dir / "database" / "yourllm.db"

    @property
    def models_cache_dir(self) -> Path:
        """Absolute path to the persistent Hugging Face cache directory."""
        return self.data_dir / "models" / "huggingface"

    def redacted_dict(self) -> dict[str, Any]:
        """Return a dictionary of configuration values with all secrets safely masked."""
        data = self.model_dump(by_alias=True)
        for key, val in data.items():
            if isinstance(getattr(self, self._field_name_for_alias(key)), SecretStr):
                data[key] = "[REDACTED]" if val else None
            elif key in {"HF_TOKEN", "YLLM_ADMIN_TOKEN"}:
                data[key] = "[REDACTED]" if val else None
        return data

    def _field_name_for_alias(self, alias: str) -> str:
        for name, field in self.model_fields.items():
            if field.alias == alias or name == alias:
                return name
        return alias


def load_settings(**kwargs: Any) -> Settings:
    """Load and validate settings, raising clean actionable errors on failure."""
    try:
        return Settings(**kwargs)
    except Exception as exc:
        # Re-raise with clean, actionable message
        raise ValueError(f"Configuration error: {exc}") from exc
