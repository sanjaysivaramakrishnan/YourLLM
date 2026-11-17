"""Unit tests for nvidia-smi execution and output parsing."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from yourllm.core.exceptions import DriverError, NvidiaSmiMissingError
from yourllm.hardware.nvidia import (
    parse_nvidia_smi_output,
    query_gpus,
)

FIXTURE_SINGLE_GPU = (
    "NVIDIA GeForce RTX 4090, GPU-a1b2c3d4-e5f6, 24564, 1024, 23540, 535.129.03, 12, 45, 115.5, 8.9"
)

FIXTURE_MULTI_GPU = (
    "NVIDIA A100-SXM4-80GB, GPU-1111-2222, 81920, 4096, 77824, 535.129.03, 25, 52, 210.0, 8.0\n"
    "NVIDIA A100-SXM4-80GB, GPU-3333-4444, 81920, 8192, 73728, 535.129.03, 15, 48, 195.0, 8.0"
)

FIXTURE_NA_FIELDS = (
    "Tesla T4, GPU-5555-6666, 15360, [N/A], 15000, 525.60.13, [Not Supported], [N/A], [N/A], 7.5"
)

FIXTURE_SHORT_LINE = (
    "NVIDIA GeForce RTX 3080, GPU-7777, 10240, 512, 9728, 535.54.03"
)


def test_parse_single_gpu() -> None:
    devices = parse_nvidia_smi_output(FIXTURE_SINGLE_GPU)
    assert len(devices) == 1
    gpu = devices[0]
    assert gpu.index == 0
    assert gpu.name == "NVIDIA GeForce RTX 4090"
    assert gpu.uuid == "GPU-a1b2c3d4-e5f6"
    assert gpu.memory_total_mib == 24564
    assert gpu.memory_used_mib == 1024
    assert gpu.memory_free_mib == 23540
    assert gpu.driver_version == "535.129.03"
    assert gpu.utilization_gpu_percent == 12.0
    assert gpu.temperature_gpu_celsius == 45
    assert gpu.power_draw_watts == 115.5
    assert gpu.compute_cap == "8.9"
    assert gpu.memory_total_gb == round(24564 / 1024.0, 2)
    assert gpu.memory_free_gb == round(23540 / 1024.0, 2)


def test_parse_multi_gpu() -> None:
    devices = parse_nvidia_smi_output(FIXTURE_MULTI_GPU)
    assert len(devices) == 2
    assert devices[0].index == 0
    assert devices[0].name == "NVIDIA A100-SXM4-80GB"
    assert devices[0].memory_total_mib == 81920
    assert devices[1].index == 1
    assert devices[1].uuid == "GPU-3333-4444"
    assert devices[1].memory_free_mib == 73728


def test_parse_na_and_unsupported_fields() -> None:
    devices = parse_nvidia_smi_output(FIXTURE_NA_FIELDS)
    assert len(devices) == 1
    gpu = devices[0]
    assert gpu.name == "Tesla T4"
    assert gpu.memory_used_mib == 0  # [N/A] safely defaults to 0
    assert gpu.memory_free_mib == 15000
    assert gpu.utilization_gpu_percent is None  # [Not Supported] -> None
    assert gpu.temperature_gpu_celsius is None  # [N/A] -> None
    assert gpu.power_draw_watts is None  # [N/A] -> None
    assert gpu.compute_cap == "7.5"


def test_parse_short_line_gracefully() -> None:
    devices = parse_nvidia_smi_output(FIXTURE_SHORT_LINE)
    assert len(devices) == 1
    gpu = devices[0]
    assert gpu.name == "NVIDIA GeForce RTX 3080"
    assert gpu.utilization_gpu_percent is None
    assert gpu.compute_cap is None


def test_parse_empty_and_whitespace() -> None:
    assert parse_nvidia_smi_output("") == []
    assert parse_nvidia_smi_output("   \n\n  \t  \n") == []


def test_parse_malformed_rows_skipped() -> None:
    malformed = "too,few,fields\nNVIDIA GPU, GPU-1, 8000, 1000, 7000, 535.10\n,,"
    devices = parse_nvidia_smi_output(malformed)
    assert len(devices) == 1
    assert devices[0].uuid == "GPU-1"


@patch("subprocess.run")
def test_query_gpus_success(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=FIXTURE_SINGLE_GPU,
        stderr="",
    )
    devices = query_gpus()
    assert len(devices) == 1
    assert devices[0].name == "NVIDIA GeForce RTX 4090"
    mock_run.assert_called_once()
    # Verify fixed arguments, no shell
    called_args, called_kwargs = mock_run.call_args
    assert called_kwargs.get("shell") is False
    assert called_args[0][0] == "nvidia-smi"


@patch("subprocess.run")
def test_query_gpus_missing_binary(mock_run: MagicMock) -> None:
    mock_run.side_effect = FileNotFoundError("No such file or directory")
    with pytest.raises(NvidiaSmiMissingError) as exc_info:
        query_gpus()
    assert "nvidia-smi was not found" in exc_info.value.message


@patch("subprocess.run")
def test_query_gpus_timeout(mock_run: MagicMock) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=5.0)
    with pytest.raises(DriverError) as exc_info:
        query_gpus(timeout=5.0)
    assert "timed out after 5.0 seconds" in exc_info.value.message


@patch("subprocess.run")
def test_query_gpus_nonzero_exit(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="NVML: Driver/library version mismatch",
    )
    with pytest.raises(DriverError) as exc_info:
        query_gpus()
    assert "Driver/library version mismatch" in exc_info.value.message
