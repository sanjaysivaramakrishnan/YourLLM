"""Tests for Pydantic Settings configuration in yourllm.core.config."""

import pytest
from pydantic import ValidationError

from yourllm.core.config import Settings, load_settings


def test_config_defaults() -> None:
    """Verify default values and alias resolution."""
    cfg = Settings(YLLM_MODEL="Qwen/Qwen2.5-7B-Instruct")
    assert cfg.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.served_model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.profile == "auto"
    assert cfg.port == 8000
    assert cfg.gpu_memory_utilization == 0.90
    assert cfg.dtype == "auto"
    assert cfg.tensor_parallel_size == 1
    assert cfg.enable_prefix_caching is True
    assert cfg.kv_cache_dtype == "auto"
    assert cfg.trust_remote_code is False
    assert cfg.max_concurrent_requests == 16
    assert cfg.request_timeout_s == 600.0
    assert cfg.max_request_bytes == 10 * 1024 * 1024
    assert cfg.log_level == "INFO"
    assert cfg.hf_token is None
    assert cfg.admin_token is None


def test_config_served_model_name_override() -> None:
    """Verify explicit served_model_name overrides default model name."""
    cfg = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1",
    )
    assert cfg.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.served_model_name == "llama-3.1"


def test_config_invalid_port() -> None:
    """Port outside 1-65535 must fail validation."""
    with pytest.raises(ValidationError) as exc:
        Settings(YLLM_MODEL="test", YLLM_PORT=70000)
    assert "Port must be between 1 and 65535" in str(exc.value)

    with pytest.raises(ValidationError):
        Settings(YLLM_MODEL="test", YLLM_PORT=0)


def test_config_invalid_gpu_memory() -> None:
    """GPU memory utilization must be in range (0.0, 1.0]."""
    with pytest.raises(ValidationError) as exc:
        Settings(YLLM_MODEL="test", YLLM_GPU_MEMORY_UTILIZATION=0.0)
    assert "YLLM_GPU_MEMORY_UTILIZATION must be in range (0.0, 1.0]" in str(exc.value)

    with pytest.raises(ValidationError):
        Settings(YLLM_MODEL="test", YLLM_GPU_MEMORY_UTILIZATION=1.1)


def test_config_invalid_tensor_parallel() -> None:
    """Tensor parallel must be >= 1."""
    with pytest.raises(ValidationError) as exc:
        Settings(YLLM_MODEL="test", YLLM_TENSOR_PARALLEL_SIZE=0)
    assert "YLLM_TENSOR_PARALLEL_SIZE must be >= 1" in str(exc.value)


def test_config_invalid_dtype() -> None:
    """Invalid dtype must be rejected."""
    with pytest.raises(ValidationError) as exc:
        Settings(YLLM_MODEL="test", YLLM_DTYPE="int4_invalid")
    assert "YLLM_DTYPE must be one of" in str(exc.value)


def test_config_invalid_log_level() -> None:
    """Invalid log level must be rejected."""
    with pytest.raises(ValidationError) as exc:
        Settings(YLLM_MODEL="test", YLLM_LOG_LEVEL="VERBOSE")
    assert "YLLM_LOG_LEVEL must be one of" in str(exc.value)


def test_config_secrets_redacted() -> None:
    """Secrets must be masked in repr and redacted_dict()."""
    secret_token = "hf_super_secret_plant_token_12345"
    admin_token = "admin_super_secret_password_67890"

    cfg = Settings(
        YLLM_MODEL="test-model",
        HF_TOKEN=secret_token,
        YLLM_ADMIN_TOKEN=admin_token,
    )

    # __repr__ and str() should mask SecretStr
    repr_str = repr(cfg)
    assert secret_token not in repr_str
    assert admin_token not in repr_str
    assert "**********" in repr_str

    # redacted_dict must show [REDACTED]
    redacted = cfg.redacted_dict()
    assert redacted["HF_TOKEN"] == "[REDACTED]"
    assert redacted["YLLM_ADMIN_TOKEN"] == "[REDACTED]"
    assert secret_token not in str(redacted)
    assert admin_token not in str(redacted)


def test_load_settings_wrapper() -> None:
    """load_settings wrapper provides clean actionable errors."""
    with pytest.raises(ValueError, match="Configuration error:"):
        load_settings(YLLM_MODEL="")

