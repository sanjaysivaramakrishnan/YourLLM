"""NVIDIA GPU query and output parsing via nvidia-smi."""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass

from yourllm.core.exceptions import DriverError, NvidiaSmiMissingError

QUERY_FIELDS = [
    "name",
    "uuid",
    "memory.total",
    "memory.used",
    "memory.free",
    "driver_version",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "compute_cap",
]


@dataclass(frozen=True)
class GPUDevice:
    """Structured representation of an NVIDIA GPU device."""

    index: int
    name: str
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    driver_version: str
    utilization_gpu_percent: float | None = None
    temperature_gpu_celsius: int | None = None
    power_draw_watts: float | None = None
    compute_cap: str | None = None

    @property
    def memory_total_gb(self) -> float:
        """Total VRAM converted to gigabytes (GiB)."""
        return round(self.memory_total_mib / 1024.0, 2)

    @property
    def memory_free_gb(self) -> float:
        """Free VRAM converted to gigabytes (GiB)."""
        return round(self.memory_free_mib / 1024.0, 2)

    @property
    def memory_used_gb(self) -> float:
        """Used VRAM converted to gigabytes (GiB)."""
        return round(self.memory_used_mib / 1024.0, 2)


def _parse_int(value: str) -> int:
    """Parse integer value handling [N/A] or [Not Supported]."""
    cleaned = value.strip()
    if not cleaned or cleaned.startswith("[") or cleaned.lower() == "n/a":
        return 0
    try:
        return int(float(cleaned))
    except ValueError:
        return 0


def _parse_float(value: str) -> float | None:
    """Parse optional float value handling [N/A] or [Not Supported]."""
    cleaned = value.strip()
    if not cleaned or cleaned.startswith("[") or cleaned.lower() == "n/a":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int_optional(value: str) -> int | None:
    """Parse optional integer value handling [N/A] or [Not Supported]."""
    cleaned = value.strip()
    if not cleaned or cleaned.startswith("[") or cleaned.lower() == "n/a":
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def parse_nvidia_smi_output(output: str) -> list[GPUDevice]:
    """Parse raw CSV output from nvidia-smi query into a list of GPUDevice objects.

    Args:
        output: Raw stdout from nvidia-smi --format=csv,noheader,nounits.

    Returns:
        List of parsed GPUDevice instances.
    """
    cleaned_output = output.strip()
    if not cleaned_output:
        return []

    devices: list[GPUDevice] = []
    reader = csv.reader(io.StringIO(cleaned_output))

    for idx, row in enumerate(reader):
        fields = [field.strip() for field in row]
        if not fields or not any(fields):
            continue

        # Expect at least name, uuid, total, used, free, driver_version
        if len(fields) < 6:
            continue

        name = fields[0]
        uuid = fields[1]
        memory_total = _parse_int(fields[2])
        memory_used = _parse_int(fields[3])
        memory_free = _parse_int(fields[4])
        driver_version = fields[5]

        utilization = _parse_float(fields[6]) if len(fields) > 6 else None
        temperature = _parse_int_optional(fields[7]) if len(fields) > 7 else None
        power_draw = _parse_float(fields[8]) if len(fields) > 8 else None

        compute_cap: str | None = None
        if len(fields) > 9:
            raw_cc = fields[9].strip()
            if raw_cc and not raw_cc.startswith("[") and raw_cc.lower() != "n/a":
                compute_cap = raw_cc

        devices.append(
            GPUDevice(
                index=idx,
                name=name,
                uuid=uuid,
                memory_total_mib=memory_total,
                memory_used_mib=memory_used,
                memory_free_mib=memory_free,
                driver_version=driver_version,
                utilization_gpu_percent=utilization,
                temperature_gpu_celsius=temperature,
                power_draw_watts=power_draw,
                compute_cap=compute_cap,
            )
        )

    return devices


def query_gpus(
    binary_path: str = "nvidia-smi",
    timeout: float = 5.0,
) -> list[GPUDevice]:
    """Execute nvidia-smi with fixed argument list and parse detected GPUs.

    Args:
        binary_path: Executable path for nvidia-smi (default "nvidia-smi").
        timeout: Subprocess timeout in seconds.

    Returns:
        List of detected GPUDevice instances.

    Raises:
        NvidiaSmiMissingError: If nvidia-smi binary cannot be executed.
        DriverError: If nvidia-smi times out or exits with non-zero code.
    """
    args = [
        binary_path,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]

    try:
        proc = subprocess.run(
            args,
            shell=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise NvidiaSmiMissingError(
            f"nvidia-smi was not found at '{binary_path}'."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DriverError(
            f"nvidia-smi query timed out after {timeout} seconds."
        ) from exc
    except Exception as exc:
        raise DriverError(
            f"Failed to execute nvidia-smi: {exc}"
        ) from exc

    if proc.returncode != 0:
        stderr_msg = proc.stderr.strip() if proc.stderr else f"Exit code {proc.returncode}"
        raise DriverError(
            f"nvidia-smi failed with code {proc.returncode}: {stderr_msg}"
        )

    return parse_nvidia_smi_output(proc.stdout)

