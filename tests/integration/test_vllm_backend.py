"""Integration tests for the VLLMBackend process supervision and readiness polling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from yourllm.core.config import Settings
from yourllm.core.exceptions import (
    InferenceCrashError,
    InferenceStartupTimeoutError,
)
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.inference.vllm_backend import VLLMBackend
from yourllm.models.registry import ResolvedModelConfig


@pytest.fixture
def test_model_config() -> ResolvedModelConfig:
    return ResolvedModelConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        served_model_name="llama-3.1-8b",
        profile_name="llama-3.1-8b-instruct",
        profile_status="untested",
        quantization=None,
        tool_call_parser="llama3_json",
        chat_template=None,
        max_model_len=4096,
        min_total_vram_gb=16,
        extra_vllm_args=[],
    )


@pytest.mark.asyncio
async def test_vllm_backend_successful_readiness(
    tmp_path: Path, test_model_config: ResolvedModelConfig
) -> None:
    """Verify backend starts process, polls readiness, and transitions to READY."""
    port = 8123
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1-8b",
        YLLM_DATA_DIR=tmp_path,
        YLLM_INTERNAL_VLLM_PORT=port,
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.STARTING_INFERENCE)

    # Harmless Python script that runs an HTTP server serving /health and /v1/models
    server_script = f"""
import http.server, json

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({{'data': [{{'id': 'llama-3.1-8b'}}]}}).encode())
        else:
            self.send_response(404)
            self.end_headers()

server = http.server.HTTPServer(('127.0.0.1', {port}), Handler)
server.serve_forever()
"""

    backend = VLLMBackend(
        settings=settings,
        model_config=test_model_config,
        lifecycle=lifecycle,
        vllm_executable=[sys.executable, "-u", "-c", server_script],
        ready_timeout_s=10.0,
        poll_interval_s=0.2,
    )

    try:
        await backend.start()
        assert backend.is_ready is True
        assert lifecycle.state == LifecycleState.READY

        # Health probe check
        healthy = await backend.check_health()
        assert healthy is True
    finally:
        await backend.stop()
        assert backend.is_ready is False


@pytest.mark.asyncio
async def test_vllm_backend_startup_crash(
    tmp_path: Path, test_model_config: ResolvedModelConfig
) -> None:
    """Verify that a child process crash during startup raises InferenceCrashError."""
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_DATA_DIR=tmp_path,
        YLLM_INTERNAL_VLLM_PORT=8124,
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.STARTING_INFERENCE)

    # Script that prints a simulated error and terminates with exit code 1
    crash_script = "import sys; sys.stderr.write('CUDA out of memory error\\n'); sys.exit(1)"

    backend = VLLMBackend(
        settings=settings,
        model_config=test_model_config,
        lifecycle=lifecycle,
        vllm_executable=[sys.executable, "-u", "-c", crash_script],
        ready_timeout_s=5.0,
        poll_interval_s=0.1,
    )

    with pytest.raises(InferenceCrashError) as exc_info:
        await backend.start()

    assert exc_info.value.code == "inference_process_crash"
    assert any("CUDA out of memory error" in line for line in exc_info.value.recent_logs)
    assert lifecycle.state == LifecycleState.FAILED


@pytest.mark.asyncio
async def test_vllm_backend_startup_timeout(
    tmp_path: Path, test_model_config: ResolvedModelConfig
) -> None:
    """Verify that a hung child process exceeds timeout and raises InferenceStartupTimeoutError."""
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_DATA_DIR=tmp_path,
        YLLM_INTERNAL_VLLM_PORT=8125,
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.STARTING_INFERENCE)

    # Script that runs silently without starting an HTTP server
    sleep_script = "import time; time.sleep(30)"

    backend = VLLMBackend(
        settings=settings,
        model_config=test_model_config,
        lifecycle=lifecycle,
        vllm_executable=[sys.executable, "-u", "-c", sleep_script],
        ready_timeout_s=0.5,
        poll_interval_s=0.1,
    )

    with pytest.raises(InferenceStartupTimeoutError) as exc_info:
        await backend.start()

    assert exc_info.value.code == "inference_startup_timeout"
    assert lifecycle.state == LifecycleState.FAILED
    assert backend.is_ready is False
