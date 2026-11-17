"""Unit tests for the lifecycle state machine and bounded recovery policy."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from yourllm.core.exceptions import (
    InvalidStateTransitionError,
    RestartLimitExceededError,
)
from yourllm.core.lifecycle import (
    LifecycleManager,
    LifecycleState,
)


def test_standard_startup_transition_chain() -> None:
    mgr = LifecycleManager()
    assert mgr.state == LifecycleState.STARTING
    assert mgr.is_ready is False

    mgr.transition_to(LifecycleState.DETECTING_GPU)
    assert mgr.state == LifecycleState.DETECTING_GPU
    assert mgr.previous_state == LifecycleState.STARTING

    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    mgr.transition_to(LifecycleState.DOWNLOADING_MODEL)
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.INITIALIZING_GATEWAY)
    mgr.transition_to(LifecycleState.READY)

    assert mgr.state == LifecycleState.READY
    assert mgr.is_ready is True
    assert mgr.is_terminal is False

    mgr.transition_to(LifecycleState.STOPPING)
    mgr.transition_to(LifecycleState.STOPPED)
    assert mgr.state == LifecycleState.STOPPED
    assert mgr.is_terminal is True


def test_cached_model_transition_shortcut() -> None:
    mgr = LifecycleManager()
    mgr.transition_to(LifecycleState.DETECTING_GPU)
    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    # Skip DOWNLOADING_MODEL directly to LOADING_MODEL
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.READY)
    assert mgr.state == LifecycleState.READY


def test_illegal_transitions_raise() -> None:
    mgr = LifecycleManager()
    # Cannot jump from STARTING to READY
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        mgr.transition_to(LifecycleState.READY)
    assert "Invalid lifecycle transition from 'STARTING' to 'READY'" in exc_info.value.message

    # Transition to READY normally
    mgr.transition_to(LifecycleState.DETECTING_GPU)
    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.READY)

    # Cannot transition from READY backwards to DETECTING_GPU
    with pytest.raises(InvalidStateTransitionError):
        mgr.transition_to(LifecycleState.DETECTING_GPU)


def test_bounded_restart_policy() -> None:
    mgr = LifecycleManager(max_restarts=3, restart_window_seconds=600)
    # Reach READY
    mgr.transition_to(LifecycleState.DETECTING_GPU)
    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.READY)

    # Crash 1
    mgr.record_process_crash("CUDA Out of Memory in batch 1")
    assert mgr.state == LifecycleState.STARTING_INFERENCE
    mgr.transition_to(LifecycleState.READY)

    # Crash 2
    mgr.record_process_crash("Process killed by OS signal")
    assert mgr.state == LifecycleState.STARTING_INFERENCE
    mgr.transition_to(LifecycleState.READY)

    # Crash 3 (max allowable restart)
    mgr.record_process_crash("Engine worker thread failed")
    assert mgr.state == LifecycleState.STARTING_INFERENCE
    mgr.transition_to(LifecycleState.READY)

    # Crash 4: Exceeds 3 restarts in 10 minutes -> transitions to FAILED
    with pytest.raises(RestartLimitExceededError) as exc_info:
        mgr.record_process_crash("Unrecoverable engine failure")

    assert mgr.state == LifecycleState.FAILED
    assert mgr.is_terminal is True
    assert "Crash limit exceeded" in str(mgr.last_error)
    assert exc_info.value.crash_count == 4


def test_restart_window_pruning() -> None:
    mgr = LifecycleManager(max_restarts=2, restart_window_seconds=100)
    mgr.transition_to(LifecycleState.DETECTING_GPU)
    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.READY)

    current_time = 1000.0
    with patch("time.monotonic", side_effect=lambda: current_time):
        mgr.record_process_crash("Old crash 1")
        mgr.transition_to(LifecycleState.READY)
        mgr.record_process_crash("Old crash 2")
        mgr.transition_to(LifecycleState.READY)

    # Advance time past window (150s later)
    current_time = 1150.0
    with patch("time.monotonic", side_effect=lambda: current_time):
        # Old crashes should be pruned, this should be crash 1 in the new window
        mgr.record_process_crash("New crash after window")
        assert mgr.state == LifecycleState.STARTING_INFERENCE


@pytest.mark.asyncio
async def test_wait_for_immediate_match() -> None:
    mgr = LifecycleManager()
    state = await mgr.wait_for(LifecycleState.STARTING, timeout=1.0)
    assert state == LifecycleState.STARTING


@pytest.mark.asyncio
async def test_wait_for_background_transition() -> None:
    mgr = LifecycleManager()

    async def _transition_later() -> None:
        await asyncio.sleep(0.05)
        mgr.transition_to(LifecycleState.DETECTING_GPU)

    asyncio.create_task(_transition_later())
    reached = await mgr.wait_for(LifecycleState.DETECTING_GPU, timeout=1.0)
    assert reached == LifecycleState.DETECTING_GPU


@pytest.mark.asyncio
async def test_wait_for_timeout() -> None:
    mgr = LifecycleManager()
    with pytest.raises(TimeoutError):
        await mgr.wait_for(LifecycleState.READY, timeout=0.05)


def test_snapshot() -> None:
    mgr = LifecycleManager()
    mgr.transition_to(LifecycleState.DETECTING_GPU, message="Scanning PCI bus")
    snap = mgr.snapshot()

    assert snap.state == "DETECTING_GPU"
    assert snap.previous_state == "STARTING"
    assert snap.total_transitions == 1
    assert len(snap.recent_transitions) == 1
    assert snap.recent_transitions[0]["from_state"] == "STARTING"
    assert snap.recent_transitions[0]["to_state"] == "DETECTING_GPU"
    assert snap.recent_transitions[0]["message"] == "Scanning PCI bus"


def test_shutdown_is_idempotent() -> None:
    mgr = LifecycleManager()
    mgr.transition_to(LifecycleState.STOPPING)
    assert mgr.state == LifecycleState.STOPPING

    # Redundant STOPPING must be a safe no-op
    mgr.transition_to(LifecycleState.STOPPING)
    assert mgr.state == LifecycleState.STOPPING

    mgr.transition_to(LifecycleState.STOPPED)
    assert mgr.state == LifecycleState.STOPPED

    # Redundant STOPPED must be a safe no-op
    mgr.transition_to(LifecycleState.STOPPED)
    assert mgr.state == LifecycleState.STOPPED

    # Attempting to STOPPING when already STOPPED is also a safe no-op
    mgr.transition_to(LifecycleState.STOPPING)
    assert mgr.state == LifecycleState.STOPPED


def test_shutdown_process_crash_race_prevention() -> None:
    mgr = LifecycleManager()
    mgr.transition_to(LifecycleState.STOPPING)

    # Process exit during STOPPING must not trigger restart or raise
    mgr.record_process_crash("Process killed via SIGTERM during shutdown")
    assert mgr.state == LifecycleState.STOPPING

    mgr.transition_to(LifecycleState.STOPPED)
    mgr.record_process_crash("Process exit after container stop")
    assert mgr.state == LifecycleState.STOPPED


@pytest.mark.asyncio
async def test_concurrent_wait_for_callers() -> None:
    mgr = LifecycleManager()

    async def _waiter() -> LifecycleState:
        return await mgr.wait_for(LifecycleState.READY, timeout=2.0)

    # Launch 5 concurrent waiters
    tasks = [asyncio.create_task(_waiter()) for _ in range(5)]

    # Advance lifecycle to READY
    mgr.transition_to(LifecycleState.DETECTING_GPU)
    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.READY)

    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(r == LifecycleState.READY for r in results)


@pytest.mark.asyncio
async def test_wait_for_transient_state_already_passed() -> None:
    mgr = LifecycleManager()
    mgr.transition_to(LifecycleState.DETECTING_GPU)
    mgr.transition_to(LifecycleState.VALIDATING_HARDWARE)
    mgr.transition_to(LifecycleState.PREPARING_MODEL)
    mgr.transition_to(LifecycleState.LOADING_MODEL)
    mgr.transition_to(LifecycleState.STARTING_INFERENCE)
    mgr.transition_to(LifecycleState.READY)

    # A caller waiting for an intermediate state that has already completed resolves immediately
    res = await mgr.wait_for(LifecycleState.VALIDATING_HARDWARE, timeout=0.5)
    assert res == LifecycleState.VALIDATING_HARDWARE


def test_failed_state_cannot_reach_ready() -> None:
    mgr = LifecycleManager()
    mgr.transition_to(LifecycleState.FAILED, error="Hardware detection error")
    assert mgr.state == LifecycleState.FAILED

    # Direct transition to READY from FAILED must be rejected
    with pytest.raises(InvalidStateTransitionError):
        mgr.transition_to(LifecycleState.READY)
