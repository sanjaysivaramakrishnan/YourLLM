"""Unit tests for hardware detection, GPU validation, and storage checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yourllm.core.exceptions import (
    InsufficientGpuCountError,
    InsufficientVramError,
    LowDiskSpaceError,
    NoDevicesError,
    UnsupportedArchitectureError,
)
from yourllm.hardware.detector import (
    check_driver_compatibility,
    detect_hardware,
    evaluate_architecture,
)
from yourllm.hardware.nvidia import GPUDevice


def create_mock_gpu(
    index: int = 0,
    name: str = "NVIDIA RTX 4090",
    uuid: str = "GPU-1234",
    total_mib: int = 24564,
    free_mib: int = 24000,
    driver: str = "535.129.03",
    compute_cap: str = "8.9",
) -> GPUDevice:
    return GPUDevice(
        index=index,
        name=name,
        uuid=uuid,
        memory_total_mib=total_mib,
        memory_used_mib=total_mib - free_mib,
        memory_free_mib=free_mib,
        driver_version=driver,
        compute_cap=compute_cap,
    )


@patch("yourllm.hardware.detector.get_disk_usage")
@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_success(
    mock_query_gpus: MagicMock,
    mock_disk_usage: MagicMock,
) -> None:
    mock_query_gpus.return_value = [create_mock_gpu()]
    mock_disk_usage.return_value = (500 * (1024**3), 200 * (1024**3))

    report = detect_hardware(
        data_dir="/data",
        tensor_parallel_size=1,
        min_vram_gb=16.0,
        required_disk_gb=50.0,
    )

    assert len(report.gpus) == 1
    assert report.total_vram_mib == 24564
    assert report.free_vram_mib == 24000
    assert report.driver_version == "535.129.03"
    assert report.cuda_compatible is True
    assert report.hardware_detected is True
    assert report.cuda_driver_compatible is True
    assert report.architecture_supported is True
    # Hardware detection must never claim inference capability has been verified
    assert report.inference_verified is False
    assert report.total_vram_gb == round(24564 / 1024.0, 2)
    assert report.free_vram_gb == round(24000 / 1024.0, 2)
    assert report.disk_free_gb == 200.0


@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_no_devices(mock_query_gpus: MagicMock) -> None:
    mock_query_gpus.return_value = []
    with pytest.raises(NoDevicesError) as exc_info:
        detect_hardware()
    assert "--gpus all" in str(exc_info.value.hint)


@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_insufficient_gpu_count(mock_query_gpus: MagicMock) -> None:
    mock_query_gpus.return_value = [create_mock_gpu(index=0)]
    with pytest.raises(InsufficientGpuCountError) as exc_info:
        detect_hardware(tensor_parallel_size=2)
    assert "Requested tensor_parallel_size=2 but only 1 GPU(s) detected" in exc_info.value.message


@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_unsupported_architecture(mock_query_gpus: MagicMock) -> None:
    # Pascal GPU (CC 6.1)
    mock_query_gpus.return_value = [
        create_mock_gpu(name="GeForce GTX 1080", compute_cap="6.1")
    ]
    with pytest.raises(UnsupportedArchitectureError) as exc_info:
        detect_hardware()
    assert "below minimum required 7.0" in exc_info.value.message
    assert "Volta V100, Turing T4" in str(exc_info.value.hint)


@patch("yourllm.hardware.detector.get_disk_usage")
@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_architecture_warning_turing(
    mock_query_gpus: MagicMock,
    mock_disk_usage: MagicMock,
) -> None:
    mock_query_gpus.return_value = [
        create_mock_gpu(name="Tesla T4", compute_cap="7.5")
    ]
    mock_disk_usage.return_value = (100 * (1024**3), 50 * (1024**3))

    report = detect_hardware()
    assert len(report.warnings) >= 1
    assert any("BF16" in w and "emulate" in w for w in report.warnings)


@patch("yourllm.hardware.detector.get_disk_usage")
@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_insufficient_vram(
    mock_query_gpus: MagicMock,
    mock_disk_usage: MagicMock,
) -> None:
    # 8 GB free, but 16 GB required
    mock_query_gpus.return_value = [
        create_mock_gpu(total_mib=8192, free_mib=8000)
    ]
    mock_disk_usage.return_value = (100 * (1024**3), 50 * (1024**3))

    with pytest.raises(InsufficientVramError) as exc_info:
        detect_hardware(min_vram_gb=16.0)
    assert "below minimum requirement (16.0 GB)" in exc_info.value.message


@patch("yourllm.hardware.detector.get_disk_usage")
@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_low_disk_space(
    mock_query_gpus: MagicMock,
    mock_disk_usage: MagicMock,
) -> None:
    mock_query_gpus.return_value = [create_mock_gpu()]
    # 10 GB free on disk, 50 GB required
    mock_disk_usage.return_value = (100 * (1024**3), 10 * (1024**3))

    with pytest.raises(LowDiskSpaceError) as exc_info:
        detect_hardware(required_disk_gb=50.0)
    assert "Insufficient disk space" in exc_info.value.message


def test_driver_compatibility_checks() -> None:
    # Test valid driver
    ok, warn = check_driver_compatibility("535.129.03")
    assert ok is True
    assert warn is None

    # Test outdated driver
    ok, warn = check_driver_compatibility("470.82.01")
    assert ok is False
    assert warn is not None
    assert "below minimum requirement" in warn


def test_evaluate_architecture_edge_cases() -> None:
    gpu_none_cc = GPUDevice(
        index=0, name="Unknown GPU", uuid="GPU-0",
        memory_total_mib=16000, memory_used_mib=0, memory_free_mib=16000,
        driver_version="535.0", compute_cap=None,
    )
    supported, warn = evaluate_architecture(gpu_none_cc)
    assert supported is True
    assert warn is not None

    gpu_unparseable = GPUDevice(
        index=0, name="Unknown GPU", uuid="GPU-0",
        memory_total_mib=16000, memory_used_mib=0, memory_free_mib=16000,
        driver_version="535.0", compute_cap="not_a_number",
    )
    supported, warn = evaluate_architecture(gpu_unparseable)
    assert supported is True


@patch("yourllm.hardware.detector.get_disk_usage")
@patch("yourllm.hardware.detector.query_gpus")
def test_detect_hardware_unqueried_compute_cap_emits_diagnostic(
    mock_query_gpus: MagicMock,
    mock_disk_usage: MagicMock,
) -> None:
    mock_query_gpus.return_value = [
        create_mock_gpu(compute_cap=None)
    ]
    mock_disk_usage.return_value = (500 * (1024**3), 200 * (1024**3))

    report = detect_hardware()
    assert report.hardware_detected is True
    assert report.inference_verified is False
    assert any("compute capability is unknown or unqueried" in w for w in report.warnings)

