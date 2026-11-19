"""Unit tests verifying vLLM CLI argument generation and flag mappings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from yourllm.core.config import Settings
from yourllm.inference.vllm_backend import build_vllm_args
from yourllm.models.registry import ResolvedModelConfig


def test_build_vllm_args_defaults() -> None:
    """Verify default CLI arguments match DECISIONS.md §5."""
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_DATA_DIR=Path("/data"),
    )
    config = ResolvedModelConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        served_model_name="meta-llama/Llama-3.1-8B-Instruct",
        profile_name="llama-3.1-8b-instruct",
        profile_status="untested",
        quantization=None,
        tool_call_parser="llama3_json",
        chat_template="/opt/yourllm/templates/tool_chat_template_llama3_json.jinja",
        max_model_len=32768,
        min_total_vram_gb=20,
        extra_vllm_args=[],
    )

    args = build_vllm_args(settings=settings, config=config, host="127.0.0.1", port=8001)

    # 1. Structural guarantees: list of strings, never a single shell string
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)

    # 2. Flag mapping validations
    assert "--model" in args
    assert args[args.index("--model") + 1] == "meta-llama/Llama-3.1-8B-Instruct"

    assert "--served-model-name" in args
    assert args[args.index("--served-model-name") + 1] == "meta-llama/Llama-3.1-8B-Instruct"

    assert "--host" in args
    assert args[args.index("--host") + 1] == "127.0.0.1"

    assert "--port" in args
    assert args[args.index("--port") + 1] == "8001"

    assert "--download-dir" in args
    assert args[args.index("--download-dir") + 1] == str(Path("/data/models/huggingface"))

    assert "--gpu-memory-utilization" in args
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"

    assert "--dtype" in args
    assert args[args.index("--dtype") + 1] == "auto"

    assert "--max-model-len" in args
    assert args[args.index("--max-model-len") + 1] == "32768"

    assert "--enable-prefix-caching" in args

    assert "--tool-call-parser" in args
    assert args[args.index("--tool-call-parser") + 1] == "llama3_json"
    assert "--enable-auto-tool-choice" in args

    assert "--chat-template" in args
    assert args[args.index("--chat-template") + 1] == "/opt/yourllm/templates/tool_chat_template_llama3_json.jinja"


def test_build_vllm_args_overrides_and_tuning() -> None:
    """Verify quantization, tensor parallelism, kv cache dtype, and extra args."""
    settings = Settings(
        YLLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        YLLM_SERVED_MODEL_NAME="qwen-32b-coder",
        YLLM_DATA_DIR=Path("/data"),
        YLLM_GPU_MEMORY_UTILIZATION=0.85,
        YLLM_DTYPE="float16",
        YLLM_QUANTIZATION="awq",
        YLLM_TENSOR_PARALLEL_SIZE=2,
        YLLM_MAX_NUM_SEQS=64,
        YLLM_KV_CACHE_DTYPE="fp8",
        YLLM_TRUST_REMOTE_CODE=True,
    )
    config = ResolvedModelConfig(
        model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        served_model_name="qwen-32b-coder",
        profile_name="qwen2.5-coder-32b-instruct-awq",
        profile_status="untested",
        quantization="awq",
        tool_call_parser="hermes",
        chat_template="/opt/yourllm/templates/tool_chat_template_hermes.jinja",
        max_model_len=16384,
        min_total_vram_gb=24,
        extra_vllm_args=["--enforce-eager"],
    )

    args = build_vllm_args(settings=settings, config=config, host="127.0.0.1", port=8001)

    assert args[args.index("--served-model-name") + 1] == "qwen-32b-coder"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert args[args.index("--dtype") + 1] == "float16"
    assert args[args.index("--quantization") + 1] == "awq"
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--max-num-seqs") + 1] == "64"
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert "--trust-remote-code" in args
    assert "--enforce-eager" in args


def test_hf_token_strictly_excluded_from_command_line() -> None:
    """Security Invariant: HF_TOKEN must NEVER be placed in the CLI argument list."""
    token_str = "hf_SUPER_SECRET_TOKEN_VALUE_12345"
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        HF_TOKEN=SecretStr(token_str),
        YLLM_DATA_DIR=Path("/data"),
    )
    config = ResolvedModelConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        served_model_name="llama-3.1-8b",
        profile_name=None,
        profile_status=None,
        quantization=None,
        tool_call_parser=None,
        chat_template=None,
        max_model_len=None,
        min_total_vram_gb=None,
        extra_vllm_args=[],
    )

    args = build_vllm_args(settings=settings, config=config)

    # Token must not appear anywhere in the args list or any argument substring
    for arg in args:
        assert token_str not in arg
        assert "hf_" not in arg
        assert "token" not in arg.lower()


def test_resolve_default_vllm_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify vLLM interpreter resolution respects explicit args, env vars, and defaults."""
    from yourllm.inference.vllm_backend import resolve_default_vllm_executable

    # 1. Explicit argument overrides all
    res = resolve_default_vllm_executable("/custom/bin/python")
    assert res == ["/custom/bin/python", "-m", "vllm.entrypoints.openai.api_server"]

    # 2. Environment variable override
    monkeypatch.setenv("YLLM_VLLM_PYTHON", "/usr/bin/python3")
    res2 = resolve_default_vllm_executable(None)
    assert res2 == ["/usr/bin/python3", "-m", "vllm.entrypoints.openai.api_server"]

    # 3. Default fallback when env is unset
    monkeypatch.delenv("YLLM_VLLM_PYTHON", raising=False)
    monkeypatch.setattr("os.path.exists", lambda path: False)
    res3 = resolve_default_vllm_executable(None)
    assert res3 == ["python3", "-m", "vllm.entrypoints.openai.api_server"]


def test_build_vllm_args_local_model_path_override() -> None:
    """Verify local snapshot path overrides --model while keeping served-model-name intact."""
    settings = Settings(
        YLLM_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ",
        YLLM_SERVED_MODEL_NAME="qwen2.5-3b-awq",
        YLLM_DATA_DIR=Path("/data"),
    )
    config = ResolvedModelConfig(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        served_model_name="qwen2.5-3b-awq",
        profile_name=None,
        profile_status=None,
        quantization="awq",
        tool_call_parser=None,
        chat_template=None,
        max_model_len=16384,
        min_total_vram_gb=4,
        extra_vllm_args=[],
    )

    local_snapshot = "/data/models/huggingface/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots/1234abcd"
    args = build_vllm_args(settings=settings, config=config, model_path=local_snapshot)

    # --model points to exact local directory on disk for 100% offline startup
    assert args[args.index("--model") + 1] == local_snapshot
    # --served-model-name points to clean public name for OpenAI client compatibility
    assert args[args.index("--served-model-name") + 1] == "qwen2.5-3b-awq"



