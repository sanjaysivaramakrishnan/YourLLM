"""Unit tests for heuristic memory estimator."""

from __future__ import annotations

from yourllm.hardware.memory_estimator import (
    estimate_memory,
    estimate_parameter_count,
)

# Realistic Llama-3.1-8B config snippet
LLAMA_8B_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}

# Realistic Qwen2.5-7B config snippet
QWEN_7B_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 152064,
    "torch_dtype": "bfloat16",
}

# Realistic Qwen2.5-32B config snippet
QWEN_32B_CONFIG = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 27648,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 152064,
    "torch_dtype": "bfloat16",
}


def test_parameter_count_estimation() -> None:
    llama_params = estimate_parameter_count(LLAMA_8B_CONFIG)
    assert llama_params is not None
    # Llama 3.1 8B is ~8.03B parameters
    assert 7.8e9 < llama_params < 8.3e9

    qwen7b_params = estimate_parameter_count(QWEN_7B_CONFIG)
    assert qwen7b_params is not None
    # Qwen 2.5 7B is ~7.6B parameters
    assert 7.2e9 < qwen7b_params < 7.9e9

    qwen32b_params = estimate_parameter_count(QWEN_32B_CONFIG)
    assert qwen32b_params is not None
    # Qwen 2.5 32B is ~32.5B parameters
    assert 31e9 < qwen32b_params < 34e9


def test_plausible_ranges_llama_8b() -> None:
    est = estimate_memory(
        config=LLAMA_8B_CONFIG,
        max_model_len=8192,
        quantization=None,
        available_vram_gb=24.0,
    )
    # Weights for 8B in FP16 should be approx 15-16 GB
    assert 14.0 <= est.estimated_weights_gb <= 16.5
    # Total range should be in 16 to 22 GB for 8k context
    assert 15.0 <= est.low_vram_gb <= 22.0
    assert est.low_vram_gb <= est.high_vram_gb
    assert est.verdict in ("likely_ok", "tight")


def test_plausible_ranges_qwen_32b_awq() -> None:
    est = estimate_memory(
        config=QWEN_32B_CONFIG,
        max_model_len=16384,
        quantization="awq",
        available_vram_gb=24.0,
    )
    # In 4-bit AWQ, weights for 32B should be ~17-19 GB
    assert 16.0 <= est.estimated_weights_gb <= 20.0
    assert est.low_vram_gb <= est.high_vram_gb


def test_monotonicity_context_length() -> None:
    # A larger context must never lower the estimate
    est_4k = estimate_memory(LLAMA_8B_CONFIG, max_model_len=4096)
    est_8k = estimate_memory(LLAMA_8B_CONFIG, max_model_len=8192)
    est_16k = estimate_memory(LLAMA_8B_CONFIG, max_model_len=16384)

    assert est_4k.low_vram_gb <= est_8k.low_vram_gb <= est_16k.low_vram_gb
    assert est_4k.high_vram_gb <= est_8k.high_vram_gb <= est_16k.high_vram_gb
    assert est_4k.estimated_kv_cache_gb <= est_8k.estimated_kv_cache_gb <= est_16k.estimated_kv_cache_gb


def test_monotonicity_quantization() -> None:
    # Quantization must strictly lower the weight and total memory estimate
    est_fp16 = estimate_memory(LLAMA_8B_CONFIG, quantization=None)
    est_fp8 = estimate_memory(LLAMA_8B_CONFIG, quantization="fp8")
    est_awq = estimate_memory(LLAMA_8B_CONFIG, quantization="awq")

    assert est_awq.estimated_weights_gb < est_fp8.estimated_weights_gb < est_fp16.estimated_weights_gb
    assert est_awq.low_vram_gb < est_fp8.low_vram_gb < est_fp16.low_vram_gb


def test_verdict_boundaries() -> None:
    est = estimate_memory(LLAMA_8B_CONFIG, max_model_len=8192, available_vram_gb=48.0)
    assert est.verdict == "likely_ok"
    assert len(est.remedies) == 0

    # Tight boundary
    tight_vram = (est.low_vram_gb + est.high_vram_gb) / 2.0
    est_tight = estimate_memory(LLAMA_8B_CONFIG, max_model_len=8192, available_vram_gb=tight_vram)
    assert est_tight.verdict == "tight"
    assert len(est_tight.remedies) > 0

    # Insufficient boundary
    est_low = estimate_memory(LLAMA_8B_CONFIG, max_model_len=8192, available_vram_gb=est.low_vram_gb - 2.0)
    assert est_low.verdict == "likely_insufficient"
    assert len(est_low.remedies) > 0
    assert any("smaller model" in r.lower() or "quantization" in r.lower() for r in est_low.remedies)


def test_unknown_config_verdict() -> None:
    est = estimate_memory({}, available_vram_gb=24.0)
    assert est.verdict == "unknown"
    assert est.low_vram_gb == 0.0


def test_output_wording_never_claims_certainty() -> None:
    est = estimate_memory(LLAMA_8B_CONFIG, available_vram_gb=24.0)
    summary_text = est.summary()
    assert "Approximate" in summary_text
    assert "guarantee" not in summary_text.lower()
    assert "fits" not in summary_text.lower()
    assert "does not guarantee" in est.notes.lower()


def test_tied_embeddings_parameter_count() -> None:
    untied_config = dict(LLAMA_8B_CONFIG)
    untied_config["tie_word_embeddings"] = False
    untied_params = estimate_parameter_count(untied_config)

    tied_config = dict(LLAMA_8B_CONFIG)
    tied_config["tie_word_embeddings"] = True
    tied_params = estimate_parameter_count(tied_config)

    assert untied_params is not None and tied_params is not None
    # Tied embeddings share vocab_size * hidden_size weights
    expected_diff = LLAMA_8B_CONFIG["vocab_size"] * LLAMA_8B_CONFIG["hidden_size"]
    assert untied_params - tied_params == expected_diff


def test_tensor_parallelism_sharding() -> None:
    est_tp1 = estimate_memory(LLAMA_8B_CONFIG, tensor_parallel_size=1)
    est_tp2 = estimate_memory(LLAMA_8B_CONFIG, tensor_parallel_size=2)

    # Weights per GPU must be halved with TP=2
    assert round(est_tp1.estimated_weights_gb / 2.0, 1) == round(est_tp2.estimated_weights_gb, 1)
    assert est_tp2.low_vram_gb < est_tp1.low_vram_gb
