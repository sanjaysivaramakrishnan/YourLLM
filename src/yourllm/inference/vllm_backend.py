"""vLLM child process inference backend implementing the InferenceBackend protocol."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import httpx

from yourllm.core.config import Settings
from yourllm.core.exceptions import (
    InferenceCrashError,
    InferenceProcessError,
    InferenceStartupTimeoutError,
)
from yourllm.core.lifecycle import LEGAL_TRANSITIONS, LifecycleManager, LifecycleState
from yourllm.inference.backend import InferenceBackend
from yourllm.inference.process import VLLMProcessManager
from yourllm.models.registry import ResolvedModelConfig

logger = logging.getLogger(__name__)

# Default vLLM v0.7.3 entrypoint vector (DECISIONS.md §1 & §4)
DEFAULT_VLLM_EXECUTABLE = ["python3", "-m", "vllm.entrypoints.openai.api_server"]


def resolve_default_vllm_executable(vllm_python: str | None = None) -> list[str]:
    """Resolve the Python executable vector for the vLLM subprocess.

    Per Architecture Invariant 1 (CLAUDE.md §3), vLLM runs in its own interpreter environment
    (the base image's system Python), while the gateway runs in an isolated virtualenv (/opt/gateway).
    """
    if vllm_python:
        return [vllm_python, "-m", "vllm.entrypoints.openai.api_server"]
    if "YLLM_VLLM_PYTHON" in os.environ and os.environ["YLLM_VLLM_PYTHON"]:
        return [os.environ["YLLM_VLLM_PYTHON"], "-m", "vllm.entrypoints.openai.api_server"]
    # In Linux / Docker container, prefer the base image's system Python (/usr/bin/python3)
    if os.path.exists("/usr/bin/python3"):
        try:
            if os.path.realpath(sys.executable) != os.path.realpath("/usr/bin/python3"):
                return ["/usr/bin/python3", "-m", "vllm.entrypoints.openai.api_server"]
        except Exception:
            return ["/usr/bin/python3", "-m", "vllm.entrypoints.openai.api_server"]
    return list(DEFAULT_VLLM_EXECUTABLE)


def build_vllm_args(
    settings: Settings,
    config: ResolvedModelConfig,
    model_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> list[str]:
    """Construct command-line argument vector for vLLM v0.7.3 based on DECISIONS.md §5.

    Guarantees:
    - Never uses shell string interpolation; returns a structured list of arguments.
    - HF_TOKEN is NEVER placed into the argument list (passed solely via environment).
    - Uses loopback host 127.0.0.1 and internal port.
    - Sets persistent Hugging Face cache download directory.
    - Uses resolved local model path on disk if available to guarantee 100% offline startup.
    """
    target_model = model_path or config.model
    args: list[str] = [
        "--model",
        target_model,
        "--served-model-name",
        config.served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--download-dir",
        str(settings.models_cache_dir),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--dtype",
        settings.dtype,
    ]

    # Context length ceiling
    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])

    # Quantization
    if config.quantization:
        args.extend(["--quantization", config.quantization])

    # Tensor parallelism (single-node multi-GPU)
    if settings.tensor_parallel_size > 1:
        args.extend(["--tensor-parallel-size", str(settings.tensor_parallel_size)])

    # Max concurrent sequences per forward pass
    if settings.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(settings.max_num_seqs)])

    # Prefix caching
    if settings.enable_prefix_caching:
        args.append("--enable-prefix-caching")

    # KV cache dtype
    if settings.kv_cache_dtype and settings.kv_cache_dtype != "auto":
        args.extend(["--kv-cache-dtype", settings.kv_cache_dtype])

    # Tool calling parser & auto tool choice
    if config.tool_call_parser:
        args.extend(["--tool-call-parser", config.tool_call_parser])
        args.append("--enable-auto-tool-choice")

    # Custom chat template
    if config.chat_template:
        args.extend(["--chat-template", config.chat_template])

    # Trust remote code
    if settings.trust_remote_code:
        args.append("--trust-remote-code")

    # Extra arguments from profile
    if config.extra_vllm_args:
        args.extend(config.extra_vllm_args)

    return args


class VLLMBackend(InferenceBackend):
    """Production InferenceBackend implementation supervising a child vLLM process."""

    def __init__(
        self,
        settings: Settings,
        model_config: ResolvedModelConfig,
        model_path: str | None = None,
        process_manager: VLLMProcessManager | None = None,
        lifecycle: LifecycleManager | None = None,
        vllm_executable: list[str] | None = None,
        base_url: str | None = None,
        ready_timeout_s: float = 600.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        self.settings = settings
        self.model_config = model_config
        self.model_path = model_path
        self.lifecycle = lifecycle
        self.vllm_executable = vllm_executable or resolve_default_vllm_executable(settings.vllm_python)
        self.ready_timeout_s = ready_timeout_s
        self.poll_interval_s = poll_interval_s

        self.internal_host = "127.0.0.1"
        self.internal_port = settings.vllm_port
        self._base_url = base_url or f"http://{self.internal_host}:{self.internal_port}"

        self.process_manager = process_manager or VLLMProcessManager(
            on_unexpected_exit=self._on_process_unexpected_exit,
        )
        self._is_ready = False

    def set_model_path(self, path: str) -> None:
        """Set resolved local snapshot directory path for offline inference."""
        self.model_path = path

    @property
    def model_name(self) -> str:
        return self.model_config.served_model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _on_process_unexpected_exit(self, returncode: int | None) -> None:
        """Handle unexpected vLLM process crash detected by process manager."""
        self._is_ready = False
        logger.error("vLLM backend crash detected (exit code: %s).", returncode)
        if self.lifecycle:
            try:
                self.lifecycle.record_process_crash(error=f"vLLM exit code {returncode}")
            except Exception as exc:
                logger.error("Lifecycle crash accounting error: %s", exc)

    async def start(self) -> None:
        """Spawn vLLM child process and await verified model readiness."""
        if self.process_manager.is_running:
            raise InferenceProcessError("vLLM backend process is already running.")

        # 1. Build CLI arguments
        cmd_args = build_vllm_args(
            settings=self.settings,
            config=self.model_config,
            model_path=self.model_path,
            host=self.internal_host,
            port=self.internal_port,
        )
        full_cmd = list(self.vllm_executable) + cmd_args

        # 2. Build isolated environment dictionary
        env = dict(os.environ)
        env["HF_HOME"] = str(self.settings.models_cache_dir)
        if self.settings.hf_token:
            env["HF_TOKEN"] = self.settings.hf_token.get_secret_value()
            env["HUGGING_FACE_HUB_TOKEN"] = self.settings.hf_token.get_secret_value()

        # Enforce offline execution when model is loaded from local snapshot path
        # or when HF_HUB_OFFLINE is set in the environment
        if self.model_path or os.environ.get("HF_HUB_OFFLINE") == "1":
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"

        # 3. Transition lifecycle to STARTING_INFERENCE if legal
        if self.lifecycle and self.lifecycle.state != LifecycleState.STARTING_INFERENCE:
            allowed = LEGAL_TRANSITIONS.get(self.lifecycle.state, set())
            if LifecycleState.STARTING_INFERENCE in allowed:
                try:
                    self.lifecycle.transition_to(LifecycleState.STARTING_INFERENCE)
                except Exception as exc:
                    logger.debug("Lifecycle transition to STARTING_INFERENCE skipped: %s", exc)

        # 4. Spawn child process
        await self.process_manager.start(cmd=full_cmd, env=env)

        # 5. Wait for readiness
        try:
            await self.wait_for_ready(timeout_s=self.ready_timeout_s)
        except Exception:
            # If startup fails or times out, terminate child process
            await self.process_manager.stop(grace_period_s=5.0)
            if self.lifecycle:
                allowed = LEGAL_TRANSITIONS.get(self.lifecycle.state, set())
                if LifecycleState.FAILED in allowed:
                    try:
                        self.lifecycle.transition_to(LifecycleState.FAILED)
                    except Exception:
                        pass
            raise

    async def stop(self) -> None:
        """Gracefully stop vLLM subprocess."""
        self._is_ready = False
        await self.process_manager.stop(grace_period_s=15.0)
        if self.lifecycle and self.lifecycle.state != LifecycleState.STOPPED:
            try:
                self.lifecycle.transition_to(LifecycleState.STOPPED)
            except Exception:
                pass

    async def check_health(self) -> bool:
        """Probe vLLM /health endpoint."""
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                resp = await client.get("/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def wait_for_ready(self, timeout_s: float) -> None:
        """Poll vLLM until /health is OK and configured model appears in /v1/models."""
        start_time = time.monotonic()
        logger.info(
            "Waiting for vLLM readiness at %s (timeout: %.0fs, model: %s)...",
            self.base_url,
            timeout_s,
            self.model_name,
        )

        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
            while time.monotonic() - start_time < timeout_s:
                # Check if process died during model loading
                if not self.process_manager.is_running:
                    recent = self.process_manager.recent_logs
                    raise InferenceCrashError(
                        exit_code=self.process_manager.returncode,
                        recent_logs=recent,
                    )

                try:
                    # 1. Health probe
                    health_resp = await client.get("/health")
                    if health_resp.status_code == 200:
                        # 2. Models probe: confirm configured model is exposed
                        models_resp = await client.get("/v1/models")
                        if models_resp.status_code == 200:
                            data = models_resp.json().get("data", [])
                            model_ids = {m.get("id") for m in data if isinstance(m, dict)}
                            expected_ids = {self.model_name, self.model_config.model}
                            if self.model_path:
                                expected_ids.add(self.model_path)
                            if any(mid in expected_ids for mid in model_ids):
                                self._is_ready = True
                                logger.info(
                                    "vLLM model '%s' confirmed ready in %.1fs.",
                                    self.model_name,
                                    time.monotonic() - start_time,
                                )
                                if self.lifecycle:
                                    try:
                                        self.lifecycle.transition_to(LifecycleState.READY)
                                    except Exception as exc:
                                        logger.debug("Lifecycle transition to READY skipped: %s", exc)
                                return
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    pass
                except Exception as exc:
                    logger.debug("Readiness check polling notice: %s", exc)

                await asyncio.sleep(self.poll_interval_s)

        # Timeout reached
        recent = self.process_manager.recent_logs
        raise InferenceStartupTimeoutError(timeout_s=timeout_s, recent_logs=recent)
