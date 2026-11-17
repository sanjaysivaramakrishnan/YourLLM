"""Health and readiness check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from yourllm.core.lifecycle import LifecycleManager

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def health_live(request: Request) -> dict[str, str]:
    """Liveness probe: returns 200 with current state name while process runs.

    Per CLAUDE.md §4:
    Unauthenticated health endpoints reveal only the state name. No model name, hardware or version.
    """
    lifecycle: LifecycleManager | None = getattr(request.app.state, "lifecycle", None)
    state_str = lifecycle.state.value if lifecycle else "STARTING"
    return {"status": state_str}


@router.get("/ready")
async def health_ready(request: Request, response: Response) -> dict[str, str]:
    """Readiness probe: returns 200 only when state is READY, otherwise 503 with Retry-After.

    Per CLAUDE.md §4:
    Unauthenticated health endpoints reveal only the state name. No model name, hardware or version.
    """
    lifecycle: LifecycleManager | None = getattr(request.app.state, "lifecycle", None)
    state_str = lifecycle.state.value if lifecycle else "STARTING"
    is_ready = lifecycle.is_ready if lifecycle else False

    if is_ready:
        return {"status": state_str}

    # Not ready: return 503 with Retry-After
    response.status_code = 503
    response.headers["Retry-After"] = "10"
    return {"status": state_str}

