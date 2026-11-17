"""Startup banner presentation printed directly to stdout upon achieving READY state."""

from __future__ import annotations

import sys
from typing import Any

from yourllm.core.config import Settings
from yourllm.hardware.detector import HardwareReport


def format_startup_banner(
    settings: Settings,
    hardware_report: HardwareReport | None = None,
    bootstrap_key: str | None = None,
    state: str = "READY",
) -> str:
    """Format the operational startup banner string."""
    lines: list[str] = [
        "",
        "=" * 78,
        f"  YourLLM Inference Server — {state}",
        "=" * 78,
        f"  State:        {state}",
        f"  Model:        {settings.model}",
        f"  Served Name:  {settings.served_model_name or settings.model}",
    ]

    # Hardware summary
    if hardware_report and hardware_report.gpus:
        gpu = hardware_report.gpus[0]
        vram_info = f"{hardware_report.total_vram_gb:.1f} GB VRAM"
        driver_info = f"Driver: {hardware_report.driver_version or 'unknown'}"
        gpu_summary = f"{gpu.name} ({vram_info}, {driver_info})"
        if len(hardware_report.gpus) > 1:
            gpu_summary += f" [+ {len(hardware_report.gpus) - 1} more GPUs]"
        lines.append(f"  Hardware:     {gpu_summary}")
    else:
        lines.append("  Hardware:     No NVIDIA GPU detected or uninitialized")

    base_url = f"http://127.0.0.1:{settings.port}/v1"
    lines.append(f"  Base URL:     {base_url}")
    lines.append(f"  Health URL:   http://127.0.0.1:{settings.port}/health/ready")
    lines.append("-" * 78)

    # Key presentation: only display if generated on THIS boot
    if bootstrap_key:
        lines.extend([
            "  Bootstrap API Key (Created on first boot):",
            f"    {bootstrap_key}",
            "",
            "  Configure your coding agent (Cline / Roo Code / Continue) with:",
            f"    Base URL: {base_url}",
            f"    API Key:  {bootstrap_key}",
            f"    Model ID: {settings.served_model_name or settings.model}",
            "-" * 78,
            "  Notice: This first-boot API key is recorded in container logs.",
            "  Rotate or revoke before exposing to public networks.",
        ])
    else:
        lines.extend([
            "  API Key Authentication:",
            "    Existing keys retained. Use 'yourllm-cli key create' to generate new keys.",
        ])

    lines.extend([
        "=" * 78,
        "",
    ])
    return "\n".join(lines)


def print_startup_banner(
    settings: Settings,
    hardware_report: HardwareReport | None = None,
    bootstrap_key: str | None = None,
    state: str = "READY",
    output_stream: Any = None,
) -> None:
    """Print the startup banner directly to stdout (bypassing the structured logger)."""
    banner = format_startup_banner(
        settings=settings,
        hardware_report=hardware_report,
        bootstrap_key=bootstrap_key,
        state=state,
    )
    stream = output_stream or sys.stdout
    print(banner, file=stream, flush=True)

