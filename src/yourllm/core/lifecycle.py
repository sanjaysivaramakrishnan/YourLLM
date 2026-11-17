"""Lifecycle state machine for YourLLM inference server."""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from yourllm.core.exceptions import (
    InvalidStateTransitionError,
    RestartLimitExceededError,
)

logger = logging.getLogger(__name__)


class LifecycleState(StrEnum):
    """Lifecycle states per CLAUDE.md §3.6."""

    STARTING = "STARTING"
    DETECTING_GPU = "DETECTING_GPU"
    VALIDATING_HARDWARE = "VALIDATING_HARDWARE"
    PREPARING_MODEL = "PREPARING_MODEL"
    DOWNLOADING_MODEL = "DOWNLOADING_MODEL"
    LOADING_MODEL = "LOADING_MODEL"
    STARTING_INFERENCE = "STARTING_INFERENCE"
    INITIALIZING_GATEWAY = "INITIALIZING_GATEWAY"
    READY = "READY"
    FAILED = "FAILED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


# Explicit transition table (CLAUDE.md §3.6 & prompt M2)
LEGAL_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
    LifecycleState.STARTING: {
        LifecycleState.DETECTING_GPU,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.DETECTING_GPU: {
        LifecycleState.VALIDATING_HARDWARE,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.VALIDATING_HARDWARE: {
        LifecycleState.PREPARING_MODEL,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.PREPARING_MODEL: {
        LifecycleState.DOWNLOADING_MODEL,
        LifecycleState.LOADING_MODEL,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.DOWNLOADING_MODEL: {
        LifecycleState.LOADING_MODEL,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.LOADING_MODEL: {
        LifecycleState.STARTING_INFERENCE,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.STARTING_INFERENCE: {
        LifecycleState.INITIALIZING_GATEWAY,
        LifecycleState.READY,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.INITIALIZING_GATEWAY: {
        LifecycleState.READY,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.READY: {
        LifecycleState.STARTING_INFERENCE,  # In-flight recovery / restart
        LifecycleState.LOADING_MODEL,
        LifecycleState.FAILED,
        LifecycleState.STOPPING,
    },
    LifecycleState.FAILED: {
        LifecycleState.STARTING,            # Manual retry/restart
        LifecycleState.STOPPING,
        LifecycleState.STOPPED,
    },
    LifecycleState.STOPPING: {
        LifecycleState.STOPPED,
        LifecycleState.FAILED,
    },
    LifecycleState.STOPPED: {
        LifecycleState.STARTING,
    },
}


@dataclass(frozen=True)
class StateTransitionRecord:
    """Record of a single state transition in the lifecycle history."""

    from_state: str
    to_state: str
    timestamp: str
    error: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class LifecycleSnapshot:
    """Read-only snapshot of current lifecycle status for APIs and monitoring."""

    state: str
    previous_state: str | None
    last_error: str | None
    uptime_seconds: float
    total_transitions: int
    recent_transitions: list[dict[str, Any]]
    restart_count: int


class LifecycleManager:
    """Thread-safe and async-aware state machine governing the YourLLM process lifecycle."""

    def __init__(
        self,
        initial_state: LifecycleState = LifecycleState.STARTING,
        max_restarts: int = 3,
        restart_window_seconds: int = 600,
        history_maxlen: int = 50,
    ) -> None:
        self._lock = threading.Lock()
        self._state = initial_state
        self._previous_state: LifecycleState | None = None
        self._last_error: str | None = None
        self._start_time = time.monotonic()
        self._max_restarts = max_restarts
        self._restart_window_seconds = restart_window_seconds

        self._history: collections.deque[StateTransitionRecord] = collections.deque(
            maxlen=history_maxlen
        )
        self._crash_timestamps: collections.deque[float] = collections.deque()
        self._total_transitions = 0
        self._condition: asyncio.Condition | None = None

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state."""
        with self._lock:
            return self._state

    @property
    def previous_state(self) -> LifecycleState | None:
        """State immediately preceding the current state."""
        with self._lock:
            return self._previous_state

    @property
    def last_error(self) -> str | None:
        """Last recorded error message."""
        with self._lock:
            return self._last_error

    @property
    def is_ready(self) -> bool:
        """True if the system is fully operational and serving inference."""
        with self._lock:
            return self._state == LifecycleState.READY

    @property
    def is_terminal(self) -> bool:
        """True if in a terminal or stopped state."""
        with self._lock:
            return self._state in {LifecycleState.FAILED, LifecycleState.STOPPED}

    def _get_condition(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    def transition_to(
        self,
        target_state: LifecycleState,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        """Transition to a target state, validating against the explicit transition table.

        Args:
            target_state: Target LifecycleState.
            error: Optional error string if transitioning due to failure.
            message: Optional explanatory message.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed by LEGAL_TRANSITIONS.
        """
        with self._lock:
            # Idempotent shutdown handling
            if self._state == target_state:
                if target_state in {LifecycleState.STOPPING, LifecycleState.STOPPED}:
                    logger.debug("Idempotent transition: already in %s", target_state.value)
                    return
            if self._state == LifecycleState.STOPPED and target_state == LifecycleState.STOPPING:
                logger.debug("Idempotent transition: already STOPPED, ignoring STOPPING request")
                return

            allowed = LEGAL_TRANSITIONS.get(self._state, set())
            if target_state not in allowed:
                raise InvalidStateTransitionError(
                    current_state=self._state.value,
                    target_state=target_state.value,
                    allowed_targets={s.value for s in allowed},
                )

            from_state = self._state
            self._previous_state = from_state
            self._state = target_state
            self._total_transitions += 1

            if error is not None:
                self._last_error = error

            rec = StateTransitionRecord(
                from_state=from_state.value,
                to_state=target_state.value,
                timestamp=datetime.now(UTC).isoformat(),
                error=error,
                message=message,
            )
            self._history.append(rec)
            logger.info(
                "Lifecycle state transitioned: %s -> %s%s",
                from_state.value,
                target_state.value,
                f" (Error: {error})" if error else "",
            )

        # Notify waiting coroutines if an event loop condition exists
        if self._condition is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._notify_condition())
            except RuntimeError:
                pass

    async def _notify_condition(self) -> None:
        if self._condition is not None:
            async with self._condition:
                self._condition.notify_all()

    def record_process_crash(self, error: str) -> None:
        """Record an unexpected inference process termination and evaluate the bounded restart policy.

        Per CLAUDE.md §3.7:
        Restart at most 3 times in 10 minutes, then transition to FAILED.

        Raises:
            RestartLimitExceededError: If crash count exceeds max_restarts within the window.
        """
        with self._lock:
            # Prevent race between shutdown and process crash/restart
            if self._state in {LifecycleState.STOPPING, LifecycleState.STOPPED}:
                logger.info(
                    "Process terminated during shutdown state '%s'; ignoring crash/restart.",
                    self._state.value,
                )
                return

            now = time.monotonic()
            self._crash_timestamps.append(now)

            # Prune crashes outside the time window
            cutoff = now - self._restart_window_seconds
            while self._crash_timestamps and self._crash_timestamps[0] < cutoff:
                self._crash_timestamps.popleft()

            crash_count = len(self._crash_timestamps)
            logger.warning(
                "Inference process crash detected (crash %d/%d within %ds window): %s",
                crash_count,
                self._max_restarts,
                self._restart_window_seconds,
                error,
            )

        if crash_count > self._max_restarts:
            self.transition_to(
                LifecycleState.FAILED,
                error=f"Crash limit exceeded: {crash_count} crashes in {self._restart_window_seconds}s. {error}",
            )
            raise RestartLimitExceededError(
                max_restarts=self._max_restarts,
                window_seconds=self._restart_window_seconds,
                crash_count=crash_count,
            )

        # If within budget, transition to STARTING_INFERENCE for bounded recovery
        self.transition_to(
            LifecycleState.STARTING_INFERENCE,
            error=error,
            message=f"Attempting bounded restart ({crash_count}/{self._max_restarts})",
        )

    async def wait_for(
        self,
        target_state: LifecycleState | set[LifecycleState],
        timeout: float | None = None,
    ) -> LifecycleState:
        """Asynchronously wait until the lifecycle reaches the target state.

        Args:
            target_state: Target state or set of acceptable target states.
            timeout: Optional maximum wait time in seconds.

        Returns:
            The matching state once reached.

        Raises:
            TimeoutError: If timeout expires before the target state is reached.
        """
        targets = {target_state} if isinstance(target_state, LifecycleState) else target_state
        target_values = {t.value for t in targets}

        with self._lock:
            if self._state in targets:
                return self._state
            # Check if any target was already reached in history
            for rec in reversed(self._history):
                if rec.to_state in target_values:
                    for t in targets:
                        if t.value == rec.to_state:
                            return t

        cond = self._get_condition()

        async def _waiter() -> LifecycleState:
            async with cond:
                while True:
                    with self._lock:
                        if self._state in targets:
                            return self._state
                        for rec in reversed(self._history):
                            if rec.to_state in target_values:
                                for t in targets:
                                    if t.value == rec.to_state:
                                        return t
                    await cond.wait()

        if timeout is not None:
            return await asyncio.wait_for(_waiter(), timeout=timeout)
        return await _waiter()

    def snapshot(self) -> LifecycleSnapshot:
        """Return a serializable status snapshot for monitoring and health endpoints."""
        with self._lock:
            uptime = round(time.monotonic() - self._start_time, 2)
            history_copy = [asdict(r) for r in self._history]
            return LifecycleSnapshot(
                state=self._state.value,
                previous_state=self._previous_state.value if self._previous_state else None,
                last_error=self._last_error,
                uptime_seconds=uptime,
                total_transitions=self._total_transitions,
                recent_transitions=history_copy,
                restart_count=len(self._crash_timestamps),
            )
