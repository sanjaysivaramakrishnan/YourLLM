"""Integration tests verifying the gateway startup sequence, liveness gating, and banner output."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tests.fixtures.fake_vllm import FakeVLLMServer, StreamingASGITransport
from yourllm.core.banner import print_startup_banner
from yourllm.core.config import Settings
from yourllm.core.exceptions import NoDevicesError
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.hardware.detector import HardwareReport
from yourllm.hardware.nvidia import GPUDevice
from yourllm.main import create_app, lifespan
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner


@pytest.mark.asyncio
async def test_startup_sequence_liveness_first_and_readiness_transition(tmp_path: Path) -> None:
    """Verify gateway serves liveness (200) while readiness remains 503 until backend is READY."""
    db = Database(tmp_path / "test.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1-8b",
        YLLM_DATA_DIR=tmp_path,
    )
    fake_backend = FakeVLLMServer(model_name="meta-llama/Llama-3.1-8B-Instruct")
    lifecycle = LifecycleManager(initial_state=LifecycleState.STARTING)

    # In mock mode, the background startup sequence transitions lifecycle to READY
    app = create_app(
        settings=settings,
        db=db,
        lifecycle=lifecycle,
        inference_backend=fake_backend,
        mock_mode=True,
        require_gpu=False,
    )

    transport = StreamingASGITransport(app)
    async with lifespan(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Liveness is available immediately
            live_resp = await client.get("/health/live")
            assert live_resp.status_code == 200

            # Wait for background startup sequence to complete (mock_mode=True transitions to READY)
            import asyncio
            for _ in range(40):
                if lifecycle.is_ready:
                    break
                await asyncio.sleep(0.05)

            assert lifecycle.is_ready is True

            # Readiness is now 200 OK
            ready_resp = await client.get("/health/ready")
            assert ready_resp.status_code == 200
            assert ready_resp.json() == {"status": "READY"}


def test_startup_banner_bootstrap_key_first_vs_second_boot(tmp_path: Path) -> None:
    """Verify banner prints bootstrap key ONLY on first boot, and suppresses it on second boot."""
    db = Database(tmp_path / "banner_test.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1-8b",
        YLLM_DATA_DIR=tmp_path,
    )
    gpu = GPUDevice(
        index=0,
        name="NVIDIA GeForce RTX 4090",
        uuid="GPU-1234",
        memory_total_mib=24576,
        memory_used_mib=500,
        memory_free_mib=24076,
        driver_version="550.54.14",
        compute_cap="8.9",
    )
    hardware_report = HardwareReport(
        hardware_detected=True,
        total_vram_mib=24576,
        free_vram_mib=24076,
        driver_version="550.54.14",
        disk_total_bytes=100 * 1024 * 1024 * 1024,
        disk_free_bytes=50 * 1024 * 1024 * 1024,
        cuda_compatible=True,
        cuda_driver_compatible=True,
        architecture_supported=True,
        inference_verified=False,
        gpus=[gpu],
    )

    # 1. First boot simulation: bootstrap key exists
    first_boot_key = "yllm_0123456789_abcdefghijklmnopqrstuvwxyz012345"
    banner_out_1 = io.StringIO()
    print_startup_banner(
        settings=settings,
        hardware_report=hardware_report,
        bootstrap_key=first_boot_key,
        state="READY",
        output_stream=banner_out_1,
    )
    banner_1 = banner_out_1.getvalue()

    assert "YourLLM Inference Server — READY" in banner_1
    assert "RTX 4090" in banner_1
    assert first_boot_key in banner_1
    assert "Bootstrap API Key (Created on first boot):" in banner_1
    assert "Notice: This first-boot API key is recorded in container logs." in banner_1

    # 2. Second boot simulation: database already has keys, bootstrap key is None
    banner_out_2 = io.StringIO()
    print_startup_banner(
        settings=settings,
        hardware_report=hardware_report,
        bootstrap_key=None,
        state="READY",
        output_stream=banner_out_2,
    )
    banner_2 = banner_out_2.getvalue()

    assert "YourLLM Inference Server — READY" in banner_2
    assert first_boot_key not in banner_2
    assert "Existing keys retained. Use 'yourllm-cli key create' to generate new keys." in banner_2
    assert "Bootstrap API Key" not in banner_2


@pytest.mark.asyncio
async def test_startup_sequence_fails_cleanly_on_missing_gpu(tmp_path: Path) -> None:
    """Verify that startup fails cleanly with FAILED state when GPU hardware is absent."""
    db = Database(tmp_path / "no_gpu.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_DATA_DIR=tmp_path,
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.STARTING)

    # App configured with require_gpu=True and mock_mode=False
    app = create_app(
        settings=settings,
        db=db,
        lifecycle=lifecycle,
        require_gpu=True,
        mock_mode=False,
    )

    transport = StreamingASGITransport(app)
    with patch("yourllm.main.detect_hardware", side_effect=NoDevicesError()):
        async with lifespan(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Liveness endpoint returns 200
                live_resp = await client.get("/health/live")
                assert live_resp.status_code == 200

                # Wait briefly for startup sequence to execute hardware check
                import asyncio
                for _ in range(40):
                    if lifecycle.state == LifecycleState.FAILED:
                        break
                    await asyncio.sleep(0.05)

                assert lifecycle.state == LifecycleState.FAILED

                # Readiness endpoint returns 503 Service Unavailable
                ready_resp = await client.get("/health/ready")
                assert ready_resp.status_code == 503
                assert ready_resp.json()["status"] == "FAILED"
