"""Hardware detection and validation for NVIDIA GPUs and host storage."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yourllm.core.exceptions import (
    InsufficientGpuCountError,
    InsufficientVramError,
    LowDiskSpaceError,
    NoDevicesError,
    UnsupportedArchitectureError,
)
from yourllm.hardware.nvidia import GPUDevice, query_gpus

# Data-driven GPU architecture policy table (DECISIONS.md & CLAUDE.md §1 & §10)
# vLLM v0.7.3 requires compute capability >= 7.0
DEFAULT_ARCH_POLICIES: list[dict[str, Any]] = [
    {
        "min_cc": 0.0,
        "max_cc": 7.0,
        "supported": False,
        "warning": (
            "Compute capability below 7.0 (Kepler, Maxwell, Pascal) is not supported "
            "by vLLM v0.7.3."
        ),
    },
    {
        "min_cc": 7.0,
        "max_cc": 8.0,
        "supported": True,
        "warning": (
            "Compute capability 7.0-7.5 (Volta V100, Turing T4, RTX 20xx) lacks native BF16 tensor "
            "operations. vLLM will emulate BF16 with reduced throughput. FP16 is strongly recommended."
        ),
    },
    {
        "min_cc": 8.0,
        "max_cc": 9.0,
        "supported": True,
        "warning": None,  # Ampere (A100, RTX 30xx) / Ada Lovelace (L40S, RTX 40xx)
    },
    {
        "min_cc": 9.0,
        "max_cc": 10.0,
        "supported": True,
        "warning": None,  # Hopper (H100)
    },
]

MIN_DRIVER_LINUX = (525, 60, 13)
MIN_DRIVER_WINDOWS = (527, 41)


@dataclass
class HardwareReport:
    """Comprehensive hardware status report.

    Distinguishes hardware detection from actual inference capability:
    - hardware_detected: Confirms physical GPU presence and nvidia-smi query.
    - cuda_driver_compatible: Validates host driver meets CUDA 12.1 minor-version requirements.
    - architecture_supported: Validates compute capability >= 7.0 per vLLM v0.7.3 baseline.
    - inference_verified: Always False during hardware detection; actual inference capability
      is only verified when vLLM initializes and executes model forward passes.
    """

    gpus: list[GPUDevice]
    total_vram_mib: int
    free_vram_mib: int
    driver_version: str
    disk_total_bytes: int
    disk_free_bytes: int
    cuda_compatible: bool = True
    hardware_detected: bool = True
    cuda_driver_compatible: bool = True
    architecture_supported: bool = True
    inference_verified: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        """Total VRAM in gigabytes (GiB)."""
        return round(self.total_vram_mib / 1024.0, 2)

    @property
    def free_vram_gb(self) -> float:
        """Free VRAM in gigabytes (GiB)."""
        return round(self.free_vram_mib / 1024.0, 2)

    @property
    def disk_free_gb(self) -> float:
        """Free storage space in gigabytes (GiB)."""
        return round(self.disk_free_bytes / (1024.0**3), 2)

    @property
    def disk_total_gb(self) -> float:
        """Total storage space in gigabytes (GiB)."""
        return round(self.disk_total_bytes / (1024.0**3), 2)


def _parse_driver_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse dot-separated driver version into a comparable integer tuple."""
    parts: list[int] = []
    for part in version_str.strip().split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def check_driver_compatibility(driver_str: str) -> tuple[bool, str | None]:
    """Check driver version against minimum required for CUDA 12.1 minor compatibility."""
    driver_tuple = _parse_driver_version_tuple(driver_str)
    if not driver_tuple:
        return True, None

    is_windows = platform.system() == "Windows"
    min_tuple = MIN_DRIVER_WINDOWS if is_windows else MIN_DRIVER_LINUX
    min_str = "527.41 (Windows)" if is_windows else "525.60.13 (Linux)"

    # Compare matching prefix length
    cmp_len = min(len(driver_tuple), len(min_tuple))
    if driver_tuple[:cmp_len] < min_tuple[:cmp_len]:
        return (
            False,
            f"Host NVIDIA driver {driver_str} is below minimum requirement {min_str} for CUDA 12.1.",
        )
    return True, None


def evaluate_architecture(
    gpu: GPUDevice,
    policies: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    """Evaluate GPU compute capability against data-driven architecture policies.

    Returns:
        (is_supported, warning_message)
    """
    if gpu.compute_cap is None:
        return True, f"GPU '{gpu.name}' compute capability is unknown or unqueried; assuming supported."

    try:
        cc_val = float(gpu.compute_cap)
    except ValueError:
        return True, f"GPU '{gpu.name}' has unparseable compute capability '{gpu.compute_cap}'."

    policy_list = policies or DEFAULT_ARCH_POLICIES
    for policy in policy_list:
        min_cc: float = policy["min_cc"]
        max_cc: float | None = policy.get("max_cc")

        matches = cc_val >= min_cc
        if max_cc is not None:
            matches = matches and (cc_val < max_cc)

        if matches:
            is_supported: bool = policy.get("supported", True)
            warn_msg: str | None = policy.get("warning")
            return is_supported, warn_msg

    # If CC is above all known policies (e.g. future CC 10.0+), treat as supported
    return True, None


def get_disk_usage(path: Path | str) -> tuple[int, int]:
    """Get total and free disk space in bytes for the specified path.

    Falls back to closest existing parent directory if path does not exist.
    """
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent

    try:
        usage = shutil.disk_usage(str(target))
        return usage.total, usage.free
    except Exception:
        # Fallback if filesystem call fails
        return 0, 0


def detect_hardware(
    data_dir: Path | str = "/data",
    tensor_parallel_size: int = 1,
    min_vram_gb: float | None = None,
    required_disk_gb: float | None = None,
    binary_path: str = "nvidia-smi",
    arch_policies: list[dict[str, Any]] | None = None,
) -> HardwareReport:
    """Detect and validate NVIDIA GPU hardware and host disk capacity.

    Args:
        data_dir: Path to persistent data storage (e.g. /data).
        tensor_parallel_size: Number of GPUs required for tensor parallelism.
        min_vram_gb: Optional minimum free VRAM required in GiB.
        required_disk_gb: Optional required free disk space in GiB.
        binary_path: Executable path for nvidia-smi.
        arch_policies: Optional custom architecture policy list.

    Returns:
        HardwareReport with GPU status, VRAM, and storage details.

    Raises:
        NvidiaSmiMissingError: If nvidia-smi cannot be found.
        DriverError: If nvidia-smi fails or times out.
        NoDevicesError: If zero NVIDIA GPUs are detected.
        InsufficientGpuCountError: If detected GPUs < tensor_parallel_size.
        UnsupportedArchitectureError: If a GPU compute capability is below minimum.
        InsufficientVramError: If free VRAM is below min_vram_gb.
        LowDiskSpaceError: If available disk space is below required_disk_gb.
    """
    gpus = query_gpus(binary_path=binary_path)

    if not gpus:
        raise NoDevicesError()

    if len(gpus) < tensor_parallel_size:
        raise InsufficientGpuCountError(
            required=tensor_parallel_size,
            available=len(gpus),
        )

    warnings: list[str] = []

    # Validate architecture for the GPUs allocated to tensor parallelism
    active_gpus = gpus[:tensor_parallel_size]
    for gpu in active_gpus:
        supported, warn = evaluate_architecture(gpu, policies=arch_policies)
        if not supported:
            raise UnsupportedArchitectureError(
                device_name=gpu.name,
                compute_cap=gpu.compute_cap or "unknown",
                min_supported="7.0",
            )
        if warn:
            warnings.append(warn)

    # Validate driver version
    primary_driver = gpus[0].driver_version
    driver_ok, driver_warn = check_driver_compatibility(primary_driver)
    if not driver_ok and driver_warn:
        warnings.append(driver_warn)

    total_vram_mib = sum(g.memory_total_mib for g in active_gpus)
    free_vram_mib = sum(g.memory_free_mib for g in active_gpus)
    free_vram_gb = free_vram_mib / 1024.0

    if min_vram_gb is not None and free_vram_gb < min_vram_gb:
        raise InsufficientVramError(
            required_gb=min_vram_gb,
            available_gb=free_vram_gb,
        )

    disk_total, disk_free = get_disk_usage(data_dir)
    disk_free_gb = disk_free / (1024.0**3)

    if required_disk_gb is not None and disk_free_gb < required_disk_gb:
        raise LowDiskSpaceError(
            path=str(data_dir),
            required_gb=required_disk_gb,
            available_gb=disk_free_gb,
        )

    return HardwareReport(
        gpus=gpus,
        total_vram_mib=total_vram_mib,
        free_vram_mib=free_vram_mib,
        driver_version=primary_driver,
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        cuda_compatible=driver_ok,
        hardware_detected=True,
        cuda_driver_compatible=driver_ok,
        architecture_supported=True,
        inference_verified=False,
        warnings=warnings,
    )
