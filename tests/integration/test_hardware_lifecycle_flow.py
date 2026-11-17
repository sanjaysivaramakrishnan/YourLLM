"""Integration test connecting hardware detection, profile resolution, and lifecycle states."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from yourllm.core.config import Settings
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.hardware.detector import detect_hardware
from yourllm.hardware.memory_estimator import estimate_memory
from yourllm.hardware.nvidia import GPUDevice
from yourllm.models.registry import resolve_model_config


def mock_4090_gpu() -> GPUDevice:
    return GPUDevice(
        index=0,
        name="NVIDIA GeForce RTX 4090",
        uuid="GPU-mock-4090",
        memory_total_mib=24564,
        memory_used_mib=1024,
        memory_free_mib=23540,
        driver_version="535.129.03",
        compute_cap="8.9",
    )


@patch("yourllm.hardware.detector.get_disk_usage")
@patch("yourllm.hardware.detector.query_gpus")
def test_full_m2_hardware_and_lifecycle_pipeline(
    mock_query_gpus: MagicMock,
    mock_disk_usage: MagicMock,
    tmp_path: Path,
) -> None:
    # 1. Mock host environment
    mock_query_gpus.return_value = [mock_4090_gpu()]
    mock_disk_usage.return_value = (500 * (1024**3), 200 * (1024**3))

    # 2. Configure runtime settings
    settings = Settings(
        YLLM_MODEL="Qwen/Qwen2.5-7B-Instruct",
        YLLM_PROFILE="auto",
        YLLM_DATA_DIR=tmp_path / "data",
    )

    # 3. Resolve profile
    resolved_config = resolve_model_config(settings)
    assert resolved_config.profile_name == "qwen2.5-7b-instruct"
    assert resolved_config.tool_call_parser == "hermes"
    assert resolved_config.min_total_vram_gb == 18

    # 4. Lifecycle begins
    lifecycle = LifecycleManager()
    assert lifecycle.state == LifecycleState.STARTING

    # Step: DETECTING_GPU
    lifecycle.transition_to(LifecycleState.DETECTING_GPU)
    hw_report = detect_hardware(
        data_dir=settings.data_dir,
        tensor_parallel_size=settings.tensor_parallel_size,
        min_vram_gb=resolved_config.min_total_vram_gb,
    )
    assert len(hw_report.gpus) == 1
    assert hw_report.free_vram_gb > 20.0

    # Step: VALIDATING_HARDWARE
    lifecycle.transition_to(LifecycleState.VALIDATING_HARDWARE)
    qwen_config = {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
    }
    estimate = estimate_memory(
        config=qwen_config,
        max_model_len=resolved_config.max_model_len or 32768,
        available_vram_gb=hw_report.free_vram_gb,
    )
    assert estimate.verdict in ("likely_ok", "tight")

    # Step: PREPARING_MODEL -> DOWNLOADING_MODEL -> LOADING_MODEL
    lifecycle.transition_to(LifecycleState.PREPARING_MODEL)
    lifecycle.transition_to(LifecycleState.DOWNLOADING_MODEL)
    lifecycle.transition_to(LifecycleState.LOADING_MODEL)

    # Step: STARTING_INFERENCE -> INITIALIZING_GATEWAY -> READY
    lifecycle.transition_to(LifecycleState.STARTING_INFERENCE)
    lifecycle.transition_to(LifecycleState.INITIALIZING_GATEWAY)
    lifecycle.transition_to(LifecycleState.READY)

    assert lifecycle.is_ready is True

    # 5. Verify status snapshot
    snap = lifecycle.snapshot()
    assert snap.state == "READY"
    assert snap.previous_state == "INITIALIZING_GATEWAY"
    assert snap.total_transitions == 8
    assert snap.restart_count == 0

    # 6. Graceful shutdown flow
    lifecycle.transition_to(LifecycleState.STOPPING)
    lifecycle.transition_to(LifecycleState.STOPPED)
    assert lifecycle.is_terminal is True

