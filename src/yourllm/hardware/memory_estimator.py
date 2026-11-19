"""Heuristic memory estimation for LLM inference on NVIDIA GPUs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]


MemoryVerdict = Literal["likely_ok", "tight", "likely_insufficient", "unknown"]


@dataclass(frozen=True)
class MemoryEstimate:
    """Heuristic memory estimate for a model configuration on target hardware.

    All numerical estimates are heuristic approximations, not operational guarantees.
    The verdict 'likely_ok' indicates nominal VRAM feasibility under standard conditions;
    it does NOT guarantee successful inference, zero OOM errors under peak concurrency,
    or successful CUDA graph capture. vLLM runtime memory management is the sole authority.

    Tensor Parallelism note:
    When tensor_parallel_size > 1, weights and KV cache heads are sharded across identical
    GPUs. All returned VRAM figures represent requirements per individual GPU.
    """

    low_vram_gb: float
    high_vram_gb: float
    estimated_weights_gb: float
    estimated_kv_cache_gb: float
    available_vram_gb: float | None
    verdict: MemoryVerdict
    remedies: list[str] = field(default_factory=list)
    notes: str = (
        "Approximate heuristic estimate. CUDA graphs and runtime buffers (~0.8 - 2.0 GB) "
        "vary by batch bucket and driver state. 'likely_ok' does not guarantee inference."
    )

    def summary(self) -> str:
        """Formatted human-readable summary of the estimate."""
        avail_str = f"{self.available_vram_gb:.1f} GB" if self.available_vram_gb is not None else "Unknown"
        return (
            f"[Approximate Estimate] Expected VRAM: {self.low_vram_gb:.1f} - {self.high_vram_gb:.1f} GB "
            f"(Weights: ~{self.estimated_weights_gb:.1f} GB, KV Cache: ~{self.estimated_kv_cache_gb:.1f} GB). "
            f"Available VRAM: {avail_str}. Verdict: {self.verdict}."
        )


def fetch_model_config(
    model_id_or_path: str,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Fetch model configuration dictionary from local path or Hugging Face Hub.

    Args:
        model_id_or_path: Local directory path or Hugging Face repository ID.
        hf_token: Optional Hugging Face token for authentication.

    Returns:
        Parsed config dictionary, or empty dict if retrieval fails.
    """
    path = Path(model_id_or_path)
    # Check if local directory containing config.json
    if path.is_dir():
        config_file = path / "config.json"
        if config_file.is_file():
            try:
                with open(config_file, encoding="utf-8") as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception:
                return {}
    elif path.is_file() and path.name == "config.json":
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except Exception:
            return {}

    # Otherwise fetch from Hugging Face Hub
    if hf_hub_download is None:
        return {}

    # Try local cache first to avoid unnecessary network queries
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_id_or_path,
            filename="config.json",
            token=hf_token,
            local_files_only=True,
        )
        with open(downloaded_path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        pass

    try:
        downloaded_path = hf_hub_download(
            repo_id=model_id_or_path,
            filename="config.json",
            token=hf_token,
            local_files_only=False,
        )
        with open(downloaded_path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return {}


def estimate_parameter_count(config: dict[str, Any]) -> int | None:
    """Estimate total model parameter count from model architecture configuration."""
    num_layers = config.get("num_hidden_layers") or config.get("n_layer")
    hidden_size = config.get("hidden_size") or config.get("n_embd")
    vocab_size = config.get("vocab_size", 32000)

    if not num_layers or not hidden_size:
        return None

    num_heads = config.get("num_attention_heads") or config.get("n_head") or (hidden_size // 128)
    num_kv_heads = config.get("num_key_value_heads") or num_heads
    head_dim = config.get("head_dim") or (hidden_size // num_heads)
    intermediate_size = config.get("intermediate_size") or int(hidden_size * 2.7)

    # Embedding layer
    embedding_params = vocab_size * hidden_size

    # Attention parameters per layer: Q, K, V, O projections
    attn_q = hidden_size * (num_heads * head_dim)
    attn_k = hidden_size * (num_kv_heads * head_dim)
    attn_v = hidden_size * (num_kv_heads * head_dim)
    attn_o = (num_heads * head_dim) * hidden_size
    attn_params = attn_q + attn_k + attn_v + attn_o

    # MLP parameters per layer (SwiGLU: gate, up, down)
    mlp_params = 3 * (hidden_size * intermediate_size)

    # Layer normalization params (RMSNorm / LayerNorm)
    norm_params = 2 * hidden_size

    per_layer = attn_params + mlp_params + norm_params
    total_layer_params = num_layers * per_layer

    # LM head (untied by default in modern LLMs; tied if tie_word_embeddings is set)
    is_tied = config.get("tie_word_embeddings", False)
    lm_head_params = 0 if is_tied else (vocab_size * hidden_size)

    return int(embedding_params + total_layer_params + lm_head_params)


def estimate_memory(
    config: dict[str, Any],
    max_model_len: int = 4096,
    quantization: str | None = None,
    dtype: str = "auto",
    kv_cache_dtype: str = "auto",
    tensor_parallel_size: int = 1,
    available_vram_gb: float | None = None,
) -> MemoryEstimate:
    """Compute a heuristic VRAM memory estimate from model configuration.

    Assumptions:
    - Model weights and KV cache attention heads are sharded across tensor_parallel_size GPUs.
    - Resulting VRAM estimates represent requirements per individual GPU.
    - Runtime overhead (CUDA graphs, context buffers, PyTorch memory pool) is heuristic (0.8 - 2.0 GiB).
    - Estimation is non-blocking and purely diagnostic; startup is never rejected by this estimator.

    Args:
        config: Model config dictionary (from config.json).
        max_model_len: Sequence length ceiling.
        quantization: Quantization method (awq, gptq, fp8, etc.).
        dtype: Model weight precision.
        kv_cache_dtype: Precision for KV cache.
        tensor_parallel_size: Tensor parallel count.
        available_vram_gb: Available free VRAM on target GPU(s) in GiB.

    Returns:
        MemoryEstimate with low/high bounds, breakdown, and verdict.
    """
    total_params = estimate_parameter_count(config)
    if total_params is None:
        return MemoryEstimate(
            low_vram_gb=0.0,
            high_vram_gb=0.0,
            estimated_weights_gb=0.0,
            estimated_kv_cache_gb=0.0,
            available_vram_gb=available_vram_gb,
            verdict="unknown",
            remedies=["Verify model repository and ensure config.json is accessible."],
            notes="Could not infer model dimensions from config.json. Startup proceeding without heuristic estimate.",
        )

    # 1. Determine bytes per weight parameter
    quant_lower = (quantization or "").strip().lower()
    if quant_lower in {"awq", "gptq"} or config.get("quantization_config", {}).get("quant_method") in {"awq", "gptq"}:
        bytes_per_param = 0.5625  # 4-bit with group scales/zeros overhead
    elif quant_lower in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        bytes_per_param = 1.0625  # 8-bit with scales
    elif dtype in {"float32"}:
        bytes_per_param = 4.0
    else:
        # Default float16 / bfloat16 (2 bytes per param)
        bytes_per_param = 2.0

    weight_bytes = total_params * bytes_per_param
    # Shard weights across tensor parallel GPUs
    tp_size = max(1, tensor_parallel_size)
    weight_bytes_per_gpu = weight_bytes / tp_size
    weights_gb = weight_bytes_per_gpu / (1024.0**3)

    # 2. Compute KV cache per sequence
    num_layers = config.get("num_hidden_layers") or config.get("n_layer", 32)
    hidden_size = config.get("hidden_size") or config.get("n_embd", 4096)
    num_heads = config.get("num_attention_heads") or config.get("n_head", 32)
    num_kv_heads = config.get("num_key_value_heads") or num_heads
    head_dim = config.get("head_dim") or (hidden_size // num_heads)

    # Shard KV heads across tensor parallel GPUs
    kv_heads_per_gpu = max(1, num_kv_heads // tp_size)

    kv_bytes_per_element = 1.0 if kv_cache_dtype.startswith("fp8") else 2.0
    # KV cache per token per GPU = 2 (keys + values) * layers * kv_heads * head_dim * bytes_per_element
    kv_bytes_per_token = 2.0 * num_layers * kv_heads_per_gpu * head_dim * kv_bytes_per_element

    # Baseline KV cache for max_model_len sequence
    single_seq_kv_bytes = kv_bytes_per_token * max_model_len
    single_seq_kv_gb = single_seq_kv_bytes / (1024.0**3)

    # Low bound: weights + 1 sequence KV cache + minimal runtime overhead (0.8 GB)
    low_overhead_gb = 0.8
    low_vram_gb = round(weights_gb + single_seq_kv_gb + low_overhead_gb, 2)

    # High bound: weights + multi-sequence KV cache buffer + CUDA graph & PyTorch overhead (2.0 GB)
    high_kv_gb = single_seq_kv_gb * 4.0
    high_overhead_gb = 2.0
    high_vram_gb = round(weights_gb + high_kv_gb + high_overhead_gb, 2)

    # 3. Formulate verdict and remedy suggestions
    verdict: MemoryVerdict = "unknown"
    remedies: list[str] = []

    if available_vram_gb is not None and available_vram_gb > 0:
        if available_vram_gb >= high_vram_gb:
            verdict = "likely_ok"
        elif available_vram_gb >= low_vram_gb:
            verdict = "tight"
            remedies.append(
                f"Available VRAM ({available_vram_gb:.1f} GB) is close to the lower estimate ({low_vram_gb:.1f} GB). "
                "Consider reducing YLLM_MAX_MODEL_LEN if out-of-memory errors occur."
            )
        else:
            verdict = "likely_insufficient"
            remedies.append(
                f"Available VRAM ({available_vram_gb:.1f} GB) is below estimated requirement ({low_vram_gb:.1f} GB)."
            )
            remedies.append("Suggested remedies:")
            remedies.append("1. Choose a smaller model (e.g. 7B or 8B parameter model).")
            if quant_lower not in {"awq", "gptq"}:
                remedies.append("2. Enable 4-bit quantization (e.g. YLLM_QUANTIZATION=awq).")
            if max_model_len > 8192:
                remedies.append("3. Decrease sequence length via YLLM_MAX_MODEL_LEN (e.g. 8192).")
            if tensor_parallel_size == 1:
                remedies.append("4. Distribute across multiple GPUs via YLLM_TENSOR_PARALLEL_SIZE.")

    return MemoryEstimate(
        low_vram_gb=low_vram_gb,
        high_vram_gb=high_vram_gb,
        estimated_weights_gb=round(weights_gb, 2),
        estimated_kv_cache_gb=round(single_seq_kv_gb, 2),
        available_vram_gb=available_vram_gb,
        verdict=verdict,
        remedies=remedies,
    )

