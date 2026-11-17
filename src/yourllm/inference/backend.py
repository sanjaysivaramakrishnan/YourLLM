"""Inference backend protocol interface.

Decouples HTTP routing and gateway layers from the underlying inference engine
(e.g., vLLM subprocess in M4 or mock engine in tests).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that any inference engine implementation must satisfy."""

    @property
    def model_name(self) -> str:
        """Served model name identifier."""
        ...

    @property
    def base_url(self) -> str:
        """Internal base URL (e.g. http://127.0.0.1:8001)."""
        ...

    @property
    def is_ready(self) -> bool:
        """True if the backend has loaded weights and is ready to process queries."""
        ...

    async def start(self) -> None:
        """Initialize and start the inference engine."""
        ...

    async def stop(self) -> None:
        """Gracefully stop the inference engine."""
        ...

    async def check_health(self) -> bool:
        """Check if the backend internal server is reachable and responsive."""
        ...

