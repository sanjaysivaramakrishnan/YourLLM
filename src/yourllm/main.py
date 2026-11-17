"""FastAPI gateway application factory and process initialization."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from yourllm.api.routes import admin, chat, health, models
from yourllm.api.routes.chat import ConcurrencyLimiter
from yourllm.auth.service import ensure_bootstrap_key
from yourllm.core.banner import print_startup_banner
from yourllm.core.config import Settings, load_settings
from yourllm.core.exceptions import InternalServerError, YourLLMError
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.core.logging import setup_logging
from yourllm.hardware.detector import HardwareReport, detect_hardware
from yourllm.hardware.memory_estimator import estimate_memory, fetch_model_config
from yourllm.inference.backend import InferenceBackend
from yourllm.inference.client import InferenceClient
from yourllm.inference.vllm_backend import VLLMBackend
from yourllm.models.manager import ModelManager
from yourllm.models.registry import resolve_model_config
from yourllm.monitoring.metrics import MetricsCollector
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner

logger = logging.getLogger(__name__)

# Context variable for request correlation ID across async logs
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects and propagates unique X-Request-ID header on all requests and responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        req_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        token = request_id_ctx.set(req_id)
        request.state.request_id = req_id

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_ctx.reset(token)


async def _run_startup_sequence(app: FastAPI) -> None:
    """Execute full startup sequence asynchronously while gateway serves liveness."""
    settings: Settings = app.state.settings
    lifecycle: LifecycleManager = app.state.lifecycle
    backend: InferenceBackend | None = getattr(app.state, "inference_backend", None)

    try:
        # 1. State: DETECTING_GPU
        if lifecycle.state == LifecycleState.STARTING:
            lifecycle.transition_to(LifecycleState.DETECTING_GPU)

        try:
            report = detect_hardware(data_dir=settings.data_dir)
        except Exception as exc:
            logger.warning("Hardware detection notice: %s", exc)
            report = HardwareReport(
                gpus=[],
                total_vram_mib=0,
                free_vram_mib=0,
                driver_version="unknown",
                disk_total_bytes=0,
                disk_free_bytes=0,
                cuda_compatible=False,
                hardware_detected=False,
                cuda_driver_compatible=False,
                architecture_supported=False,
                inference_verified=False,
                warnings=[str(exc)],
            )
        app.state.hardware_report = report

        # Check for missing GPU hardware
        if not report.hardware_detected:
            if getattr(app.state, "require_gpu", True) and not getattr(app.state, "mock_mode", False):
                logger.error(
                    "No NVIDIA GPU detected. YourLLM requires an NVIDIA GPU with compute capability >= 7.0."
                )
                lifecycle.transition_to(LifecycleState.FAILED, error="No NVIDIA GPU detected")
                return

        # 2. State: VALIDATING_HARDWARE
        if lifecycle.state == LifecycleState.DETECTING_GPU:
            lifecycle.transition_to(LifecycleState.VALIDATING_HARDWARE)

        # Check compute capability
        if not report.architecture_supported and not getattr(app.state, "mock_mode", False):
            logger.error(
                "Unsupported GPU architecture detected (CC < 7.0). vLLM v0.7.3 requires CC >= 7.0."
            )
            lifecycle.transition_to(LifecycleState.FAILED, error="Unsupported GPU architecture (CC < 7.0)")
            return

        # 3. State: PREPARING_MODEL
        if lifecycle.state == LifecycleState.VALIDATING_HARDWARE:
            lifecycle.transition_to(LifecycleState.PREPARING_MODEL)

        # Model profile resolution & heuristic memory advisory
        model_config = resolve_model_config(settings)
        app.state.model_config = model_config
        for warn in model_config.warnings:
            logger.warning("Profile resolution warning: %s", warn)

        try:
            hf_token_val = settings.hf_token.get_secret_value() if settings.hf_token else None
            cfg_dict = fetch_model_config(model_config.model, hf_token=hf_token_val)
            if cfg_dict:
                mem_estimate = estimate_memory(
                    config=cfg_dict,
                    max_model_len=model_config.max_model_len or 4096,
                    quantization=model_config.quantization,
                    dtype=settings.dtype,
                    kv_cache_dtype=settings.kv_cache_dtype,
                    tensor_parallel_size=settings.tensor_parallel_size,
                    available_vram_gb=report.total_vram_gb,
                )
                if mem_estimate.verdict in {"tight", "likely_insufficient"}:
                    logger.warning(
                        "Advisory memory estimation [%s]: %s (remedies: %s)",
                        mem_estimate.verdict,
                        mem_estimate.summary(),
                        "; ".join(mem_estimate.remedies),
                    )
        except Exception as exc:
            logger.debug("Memory estimation skipped: %s", exc)

        # 4. State: DOWNLOADING_MODEL -> LOADING_MODEL
        if not getattr(app.state, "mock_mode", False):
            if lifecycle.state == LifecycleState.PREPARING_MODEL:
                lifecycle.transition_to(LifecycleState.DOWNLOADING_MODEL)
            manager = ModelManager(
                cache_dir=settings.models_cache_dir,
                hf_token=settings.hf_token,
            )
            await asyncio.to_thread(manager.prepare_model, model_config.model)
            if lifecycle.state == LifecycleState.DOWNLOADING_MODEL:
                lifecycle.transition_to(LifecycleState.LOADING_MODEL)
        else:
            if lifecycle.state == LifecycleState.PREPARING_MODEL:
                lifecycle.transition_to(LifecycleState.LOADING_MODEL)

        # 5. State: STARTING_INFERENCE
        if lifecycle.state == LifecycleState.LOADING_MODEL:
            lifecycle.transition_to(LifecycleState.STARTING_INFERENCE)

        # 6. Start supervised inference engine
        if backend is not None:
            await backend.start()
            if not lifecycle.is_ready and getattr(backend, "is_ready", False):
                lifecycle.transition_to(LifecycleState.READY)

        # 7. Print startup banner upon achieving READY
        if lifecycle.is_ready:
            print_startup_banner(
                settings=settings,
                hardware_report=getattr(app.state, "hardware_report", None),
                bootstrap_key=getattr(app.state, "bootstrap_key", None),
                state="READY",
            )

    except Exception as exc:
        logger.error("Startup sequence failed: %s", exc, exc_info=True)
        try:
            lifecycle.transition_to(LifecycleState.FAILED, error=str(exc))
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and graceful shutdown."""
    settings: Settings = app.state.settings
    db: Database | None = getattr(app.state, "db", None)
    metrics: MetricsCollector = app.state.metrics
    client: InferenceClient | None = getattr(app.state, "inference_client", None)
    lifecycle: LifecycleManager = app.state.lifecycle

    # 1. Database migrations and bootstrap key
    bootstrap_key: str | None = None
    if db is not None:
        with db.connect() as conn:
            runner = MigrationRunner(conn)
            runner.apply_pending()
        bootstrap_key = ensure_bootstrap_key(db)
        app.state.bootstrap_key = bootstrap_key

    # 2. Initialize default VLLMBackend if not injected
    if getattr(app.state, "inference_backend", None) is None:
        model_config = resolve_model_config(settings)
        app.state.model_config = model_config
        app.state.inference_backend = VLLMBackend(
            settings=settings,
            model_config=model_config,
            lifecycle=lifecycle,
        )

    # 3. Launch background startup sequence if not already READY
    startup_task: asyncio.Task[None] | None = None
    auto_start = getattr(app.state, "auto_start_backend", True)
    if auto_start and not lifecycle.is_ready:
        startup_task = asyncio.create_task(_run_startup_sequence(app))

    yield

    # 4. Shutdown tasks (Graceful termination on SIGTERM/SIGINT)
    logger.info("Gateway shutting down: stopping inference backend and releasing connections.")
    if startup_task and not startup_task.done():
        startup_task.cancel()

    backend: InferenceBackend | None = getattr(app.state, "inference_backend", None)
    if backend is not None:
        try:
            await backend.stop()
        except Exception as exc:
            logger.debug("Error stopping inference backend during shutdown: %s", exc)

    if db is not None:
        served_model = settings.served_model_name or settings.model
        metrics.flush_to_db(db, served_model)

    if client is not None:
        await client.aclose()


def create_app(
    settings: Settings | None = None,
    db: Database | None = None,
    lifecycle: LifecycleManager | None = None,
    inference_client: InferenceClient | None = None,
    inference_backend: InferenceBackend | None = None,
    metrics: MetricsCollector | None = None,
    auto_start_backend: bool = True,
    require_gpu: bool = True,
    mock_mode: bool = False,
) -> FastAPI:
    """Create and configure the YourLLM FastAPI gateway application."""
    active_settings = settings or load_settings()
    setup_logging(level=active_settings.log_level)

    app = FastAPI(
        title="YourLLM Gateway",
        description="Self-hosted, OpenAI-compatible inference server for AI coding agents.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Initialize dependencies on application state
    app.state.settings = active_settings
    app.state.db = db or Database(active_settings.database_path)
    app.state.lifecycle = lifecycle or LifecycleManager(initial_state=LifecycleState.STARTING)
    app.state.metrics = metrics or MetricsCollector()
    app.state.concurrency_limiter = ConcurrencyLimiter(
        max_concurrent=active_settings.max_concurrent_requests
    )
    app.state.auto_start_backend = auto_start_backend
    app.state.require_gpu = require_gpu
    app.state.mock_mode = mock_mode

    if inference_client is not None:
        app.state.inference_client = inference_client
    else:
        app.state.inference_client = InferenceClient(
            base_url=f"http://127.0.0.1:{active_settings.vllm_port}",
            request_timeout_s=active_settings.request_timeout_s,
        )

    if inference_backend is not None:
        app.state.inference_backend = inference_backend

    # Register middleware
    app.add_middleware(RequestIDMiddleware)

    # Register exception handlers
    @app.exception_handler(YourLLMError)
    async def yourllm_error_handler(request: Request, exc: YourLLMError) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "")
        headers = {"X-Request-ID": req_id}
        if hasattr(exc, "retry_after"):
            headers["Retry-After"] = str(exc.retry_after)

        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "")
        logger.error("Unhandled server exception [%s]: %s", req_id, exc, exc_info=True)
        fallback = InternalServerError("An unexpected internal error occurred.")
        return JSONResponse(
            status_code=500,
            content=fallback.to_dict(),
            headers={"X-Request-ID": req_id},
        )

    # Mount API routers
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    return app


def run() -> None:
    """CLI/container entrypoint launching the gateway via uvicorn."""
    import uvicorn

    settings = load_settings()
    app = create_app(settings=settings)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_config=None,  # Use YourLLM structured JSON logging
        access_log=False,
    )


if __name__ == "__main__":
    run()
