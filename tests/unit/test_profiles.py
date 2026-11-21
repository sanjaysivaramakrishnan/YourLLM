"""Unit tests for model profiles, registry loading, and config resolution."""

from __future__ import annotations

import pytest

from yourllm.core.config import Settings
from yourllm.core.exceptions import (
    ProfileNotFoundError,
    UnsupportedParserError,
)
from yourllm.models.registry import (
    PINNED_VLLM_PARSERS,
    ModelProfile,
    ProfileRegistry,
    resolve_model_config,
)


def test_shipped_profiles_validity() -> None:
    registry = ProfileRegistry()
    profiles = registry.list_all()

    # Curated shipped profiles
    assert 2 <= len(profiles) <= 6

    names = {p.name for p in profiles}
    assert "llama-3.1-8b-instruct" in names
    assert "qwen2.5-7b-instruct" in names
    assert "qwen2.5-coder-32b-instruct-awq" in names
    assert "qwen2.5-coder-1.5b-instruct" in names
    assert "qwen2.5-3b-instruct-awq" in names

    for p in profiles:
        # All shipped profiles must be initially marked "untested"
        assert p.status == "untested"
        if p.tool_call_parser:
            assert p.tool_call_parser in PINNED_VLLM_PARSERS


def test_schema_rejects_unsupported_parser() -> None:
    with pytest.raises(UnsupportedParserError) as exc_info:
        ModelProfile(
            name="invalid-parser-profile",
            hf_repo="test/repo",
            tool_call_parser="nonexistent_parser_v9",
        )
    assert "nonexistent_parser_v9" in str(exc_info.value)
    assert "Allowed parsers" in exc_info.value.message


def test_resolve_auto_matching() -> None:
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_PROFILE="auto",
    )
    resolved = resolve_model_config(settings)

    assert resolved.profile_name == "llama-3.1-8b-instruct"
    assert resolved.profile_status == "untested"
    assert resolved.tool_call_parser == "llama3_json"
    assert resolved.chat_template == "/opt/yourllm/templates/tool_chat_template_llama3_json.jinja"
    assert resolved.max_model_len == 32768
    assert resolved.min_total_vram_gb == 20
    assert len(resolved.warnings) == 0


def test_resolve_explicit_profile() -> None:
    settings = Settings(
        YLLM_MODEL="custom-path/model",
        YLLM_PROFILE="qwen2.5-7b-instruct",
    )
    resolved = resolve_model_config(settings)

    assert resolved.profile_name == "qwen2.5-7b-instruct"
    assert resolved.tool_call_parser == "hermes"


def test_resolve_explicit_profile_not_found() -> None:
    settings = Settings(
        YLLM_MODEL="custom/model",
        YLLM_PROFILE="non-existent-profile",
    )
    with pytest.raises(ProfileNotFoundError) as exc_info:
        resolve_model_config(settings)
    assert "non-existent-profile" in exc_info.value.message


def test_resolve_unprofiled_model() -> None:
    settings = Settings(
        YLLM_MODEL="unprofiled/org-model",
        YLLM_PROFILE="auto",
    )
    resolved = resolve_model_config(settings)

    assert resolved.profile_name is None
    assert resolved.profile_status is None
    assert resolved.tool_call_parser is None


def test_resolve_unprofiled_qwen_auto_infers_hermes() -> None:
    settings = Settings(
        YLLM_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ",
        YLLM_PROFILE="auto",
    )
    resolved = resolve_model_config(settings)

    assert resolved.tool_call_parser == "hermes"
    assert resolved.chat_template == "/opt/yourllm/templates/tool_chat_template_hermes.jinja"



def test_resolve_env_overrides_with_warnings() -> None:
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_PROFILE="auto",
        YLLM_TOOL_CALL_PARSER="hermes",  # Override profile's llama3_json
        YLLM_QUANTIZATION="fp8",          # Override profile's null
        YLLM_MAX_MODEL_LEN=16384,         # Override profile's 32768
    )
    resolved = resolve_model_config(settings)

    assert resolved.tool_call_parser == "hermes"
    assert resolved.quantization == "fp8"
    assert resolved.max_model_len == 16384
    assert len(resolved.warnings) >= 2
    assert any("tool_call_parser" in w for w in resolved.warnings)
    assert any("quantization" in w for w in resolved.warnings)


def test_resolve_env_override_bad_parser() -> None:
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_PROFILE="auto",
        YLLM_TOOL_CALL_PARSER="bad_parser_name",
    )
    with pytest.raises(UnsupportedParserError) as exc_info:
        resolve_model_config(settings)
    assert "bad_parser_name" in exc_info.value.message

