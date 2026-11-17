"""Unit tests for the vLLM asynchronous subprocess manager."""

from __future__ import annotations

import asyncio
import sys

import pytest

from yourllm.inference.process import VLLMProcessManager, sanitize_vllm_log


def test_sanitize_vllm_log() -> None:
    """Verify secrets are redacted from raw log lines."""
    raw = "Loading model with token hf_abcdef1234567890abcdef and key yllm_0123456789_abcdefghijklmnopqrstuvwxyz012345"
    sanitized = sanitize_vllm_log(raw)
    assert "hf_abcdef1234567890abcdef" not in sanitized
    assert "yllm_" not in sanitized
    assert "[REDACTED]" in sanitized


@pytest.mark.asyncio
async def test_process_manager_start_and_graceful_stop() -> None:
    """Verify clean subprocess spawn, log streaming, and graceful SIGTERM termination."""
    manager = VLLMProcessManager()

    # Harmless Python child that logs and sleeps
    child_cmd = [
        sys.executable,
        "-u",
        "-c",
        "import sys, time; print('Ready for inference'); sys.stdout.flush(); time.sleep(30)",
    ]

    await manager.start(cmd=child_cmd)
    assert manager.is_running is True
    assert manager.pid is not None
    assert manager.returncode is None

    # Allow stream reader to capture stdout
    for _ in range(20):
        if any("Ready for inference" in line for line in manager.recent_logs):
            break
        await asyncio.sleep(0.05)

    assert any("Ready for inference" in line for line in manager.recent_logs)

    # Stop process gracefully
    exit_code = await manager.stop(grace_period_s=5.0)
    assert manager.is_running is False
    assert exit_code is not None


@pytest.mark.asyncio
async def test_process_manager_forced_sigkill_on_hung_process() -> None:
    """Verify that a process ignoring SIGTERM is terminated via SIGKILL after grace period."""
    manager = VLLMProcessManager()

    # Harmless child that ignores SIGTERM (on POSIX) or sleeps in an uninterruptible loop
    child_code = (
        "import sys, signal, time; "
        "signal.signal(signal.SIGTERM, lambda s, f: None) if hasattr(signal, 'SIGTERM') else None; "
        "print('Started'); sys.stdout.flush(); "
        "time.sleep(30)"
    )

    await manager.start(cmd=[sys.executable, "-u", "-c", child_code])
    assert manager.is_running is True

    for _ in range(20):
        if any("Started" in line for line in manager.recent_logs):
            break
        await asyncio.sleep(0.05)

    # Stop with very short grace period (0.2s) to trigger SIGKILL fallback
    exit_code = await manager.stop(grace_period_s=0.2)
    assert manager.is_running is False
    assert exit_code is not None


@pytest.mark.asyncio
async def test_process_manager_unexpected_exit_callback() -> None:
    """Verify that an unexpected process crash triggers on_unexpected_exit callback."""
    unexpected_exit_event = asyncio.Event()
    received_code: list[int | None] = []

    def on_exit(code: int | None) -> None:
        received_code.append(code)
        unexpected_exit_event.set()

    manager = VLLMProcessManager(on_unexpected_exit=on_exit)

    # Child script that terminates immediately with non-zero exit code 42
    await manager.start(cmd=[sys.executable, "-c", "import sys; sys.exit(42)"])

    # Wait for the unexpected exit to be detected by the watcher task
    await asyncio.wait_for(unexpected_exit_event.wait(), timeout=5.0)

    assert len(received_code) == 1
    assert received_code[0] == 42
    assert manager.is_running is False

    await manager.stop()

