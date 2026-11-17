"""Administrative HTTP endpoints for operational monitoring, status, and hardware reports."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from yourllm.auth.dependencies import require_admin
from yourllm.core.config import Settings
from yourllm.core.lifecycle import LifecycleManager
from yourllm.hardware.detector import HardwareReport
from yourllm.monitoring.metrics import MetricsCollector

router = APIRouter(prefix="/admin/v1", tags=["admin"])


@router.get("/status")
async def get_admin_status(
    request: Request,
    _token: Annotated[str, Depends(require_admin)],
) -> dict[str, Any]:
    """Retrieve runtime lifecycle snapshot, operational metrics, and model identifier."""
    lifecycle: LifecycleManager | None = getattr(request.app.state, "lifecycle", None)
    metrics: MetricsCollector | None = getattr(request.app.state, "metrics", None)
    settings: Settings | None = getattr(request.app.state, "settings", None)

    lifecycle_snap = lifecycle.snapshot() if lifecycle else None
    metrics_snap = metrics.snapshot() if metrics else {}
    served_model = settings.served_model_name if settings else "unknown"

    return {
        "model": served_model,
        "lifecycle": lifecycle_snap.__dict__ if lifecycle_snap else {},
        "metrics": metrics_snap,
    }


@router.get("/hardware")
async def get_admin_hardware(
    request: Request,
    _token: Annotated[str, Depends(require_admin)],
) -> dict[str, Any]:
    """Retrieve host NVIDIA GPU and storage hardware status report."""
    hardware_report: HardwareReport | None = getattr(request.app.state, "hardware_report", None)
    if not hardware_report:
        return {"status": "uninitialized", "report": None}

    gpus_data = [
        {
            "index": g.index,
            "name": g.name,
            "uuid": g.uuid,
            "memory_total_gb": g.memory_total_gb,
            "memory_free_gb": g.memory_free_gb,
            "memory_used_gb": g.memory_used_gb,
            "driver_version": g.driver_version,
            "compute_cap": g.compute_cap,
            "temperature_gpu_celsius": g.temperature_gpu_celsius,
            "utilization_gpu_percent": g.utilization_gpu_percent,
        }
        for g in hardware_report.gpus
    ]

    return {
        "gpus": gpus_data,
        "total_vram_gb": hardware_report.total_vram_gb,
        "free_vram_gb": hardware_report.free_vram_gb,
        "driver_version": hardware_report.driver_version,
        "disk_free_gb": hardware_report.disk_free_gb,
        "disk_total_gb": hardware_report.disk_total_gb,
        "cuda_compatible": hardware_report.cuda_compatible,
        "hardware_detected": hardware_report.hardware_detected,
        "cuda_driver_compatible": hardware_report.cuda_driver_compatible,
        "architecture_supported": hardware_report.architecture_supported,
        "inference_verified": hardware_report.inference_verified,
        "warnings": hardware_report.warnings,
    }


@router.get("/config")
async def get_admin_config(
    request: Request,
    _token: Annotated[str, Depends(require_admin)],
) -> dict[str, Any]:
    """Retrieve active runtime configuration with all secrets strictly redacted."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if not settings:
        return {}
    # Guaranteed safe redaction of HF_TOKEN, YLLM_ADMIN_TOKEN, etc.
    return settings.redacted_dict()

