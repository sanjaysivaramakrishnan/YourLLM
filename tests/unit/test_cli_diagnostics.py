"""Unit tests for `yourllm-cli doctor` and `yourllm-cli bench` diagnostics."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from yourllm.cli.diagnostics import (
    BenchmarkSample,
    BenchmarkSummary,
    CheckResult,
    DiagnosticsError,
    DoctorReport,
    normalize_base_url,
    run_benchmark,
    run_doctor,
)
from yourllm.cli.main import app

runner = CliRunner()


def _sse_response(events: list[bytes], status_code: int = 200) -> httpx.Response:
    """Create a streaming-compatible mock httpx.Response from raw SSE event bytes."""
    return httpx.Response(
        status_code=status_code,
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
        content=b"".join(events),
    )


def _build_healthy_transport(
    valid_key: str = "yllm_test123_validsecret",
    model_id: str = "qwen2.5-3b-awq",
    tool_mode: str = "valid",
    include_usage: bool = True,
) -> httpx.MockTransport:
    """Build a deterministic MockTransport simulating a healthy YourLLM gateway."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        auth = request.headers.get("authorization")

        if path == "/health/live":
            return httpx.Response(200, json={"status": "READY"})

        if path == "/health/ready":
            return httpx.Response(200, json={"status": "READY"})

        if auth != f"Bearer {valid_key}":
            return httpx.Response(
                401,
                json={"error": {"message": "Unauthorized", "type": "authentication_error"}},
            )

        if path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": model_id, "object": "model"}]},
            )

        if path == "/v1/chat/completions":
            payload: dict[str, Any] = json.loads(request.content.decode())
            if payload.get("tools"):
                if tool_mode == "valid":
                    return _sse_response(
                        [
                            b'data: {"id":"cmpl-t1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"add_numbers","arguments":""}}]},"finish_reason":null}]}\n\n',
                            b'data: {"id":"cmpl-t1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\": 2, \\"b\\": 3}"}}]},"finish_reason":null}]}\n\n',
                            b'data: {"id":"cmpl-t1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
                            b"data: [DONE]\n\n",
                        ]
                    )
                if tool_mode == "broken":
                    return _sse_response(
                        [
                            b'data: {"id":"cmpl-t2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"add_numbers","arguments":"{\\"a\\":2,\\"b\\":3}"}}]},"finish_reason":null}]}\n\n',
                            b'data: {"id":"cmpl-t2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                            b"data: [DONE]\n\n",
                        ]
                    )
                # Plain-text fallback (no tool call emitted)
                return _sse_response(
                    [
                        b'data: {"id":"cmpl-t3","choices":[{"index":0,"delta":{"content":"2 + 3 = 5"},"finish_reason":null}]}\n\n',
                        b'data: {"id":"cmpl-t3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                        b"data: [DONE]\n\n",
                    ]
                )

            # Standard streaming chat completion
            final_chunk: dict[str, Any] = {
                "id": "cmpl-s1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if include_usage:
                final_chunk["usage"] = {
                    "prompt_tokens": 18,
                    "completion_tokens": 25,
                    "total_tokens": 43,
                }
            return _sse_response(
                [
                    b'data: {"id":"cmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"def "},"finish_reason":null}]}\n\n',
                    b'data: {"id":"cmpl-s1","choices":[{"index":0,"delta":{"content":"binary_search"},"finish_reason":null}]}\n\n',
                    b'data: {"id":"cmpl-s1","choices":[{"index":0,"delta":{"content":"(arr, x):"},"finish_reason":null}]}\n\n',
                    b'data: {"id":"cmpl-s1","choices":[{"index":0,"delta":{"content":" pass"},"finish_reason":null}]}\n\n',
                    f"data: {json.dumps(final_chunk)}\n\n".encode(),
                    b"data: [DONE]\n\n",
                ]
            )

        return httpx.Response(404, json={"error": {"message": "Not found"}})

    return httpx.MockTransport(handler)


def test_normalize_base_url() -> None:
    assert normalize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert normalize_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    assert normalize_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert normalize_base_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000"


def test_run_doctor_all_checks_pass() -> None:
    transport = _build_healthy_transport()
    with httpx.Client(transport=transport) as client:
        report = run_doctor(
            base_url="http://127.0.0.1:8000/v1",
            api_key="yllm_test123_validsecret",
            client=client,
        )

    assert report.all_passed is True
    assert report.warning_count == 0
    assert report.model_id == "qwen2.5-3b-awq"
    assert len(report.checks) == 6
    assert all(c.passed for c in report.checks)


def test_run_doctor_detects_broken_tool_call_metadata_and_finish_reason() -> None:
    transport = _build_healthy_transport(tool_mode="broken")
    with httpx.Client(transport=transport) as client:
        report = run_doctor(
            base_url="http://127.0.0.1:8000",
            api_key="yllm_test123_validsecret",
            client=client,
        )

    assert report.all_passed is False
    tool_check = report.checks[-1]
    assert tool_check.name == "Tool-Call Compatibility"
    assert tool_check.passed is False
    assert "id" in tool_check.detail
    assert "type='function'" in tool_check.detail
    assert "finish_reason='tool_calls'" in tool_check.detail


def test_run_doctor_warns_when_model_skips_tool_call() -> None:
    transport = _build_healthy_transport(tool_mode="plain_text")
    with httpx.Client(transport=transport) as client:
        report = run_doctor(
            base_url="http://127.0.0.1:8000",
            api_key="yllm_test123_validsecret",
            client=client,
        )

    assert report.all_passed is True
    assert report.warning_count == 1
    assert report.checks[-1].warning is True


def test_run_doctor_not_ready_returns_early() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "LOADING_MODEL"})
        return httpx.Response(503, json={"status": "LOADING_MODEL"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = run_doctor(
            base_url="http://127.0.0.1:8000",
            api_key="yllm_test123_validsecret",
            client=client,
        )

    assert report.all_passed is False
    assert len(report.checks) == 2
    assert "LOADING_MODEL" in report.checks[1].detail


def test_run_doctor_unreachable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = run_doctor(base_url="http://127.0.0.1:9999", client=client)

    assert report.all_passed is False
    assert len(report.checks) == 1
    assert "Cannot reach server" in report.checks[0].detail


def test_run_doctor_invalid_key_never_leaks_secret() -> None:
    planted_key = "yllm_secret999_DO_NOT_LEAK_PLAINTEXT"
    transport = _build_healthy_transport(valid_key="yllm_other_key")
    with httpx.Client(transport=transport) as client:
        report = run_doctor(
            base_url="http://127.0.0.1:8000",
            api_key=planted_key,
            client=client,
        )

    assert report.all_passed is False
    serialized = json.dumps(report.to_dict())
    assert planted_key not in serialized
    assert "DO_NOT_LEAK_PLAINTEXT" not in serialized


def test_run_benchmark_with_exact_usage() -> None:
    transport = _build_healthy_transport(include_usage=True)
    observed: list[BenchmarkSample] = []

    with httpx.Client(transport=transport) as client:
        summary = run_benchmark(
            base_url="http://127.0.0.1:8000",
            api_key="yllm_test123_validsecret",
            runs=3,
            max_tokens=64,
            client=client,
            on_sample=observed.append,
        )

    assert summary.model == "qwen2.5-3b-awq"
    assert summary.runs == 3
    assert len(observed) == 3
    assert summary.total_completion_tokens == 75
    assert summary.avg_tps > 0.0
    assert summary.min_ttft_ms <= summary.avg_ttft_ms <= summary.max_ttft_ms
    assert all(not s.estimated_tokens for s in summary.samples)


def test_run_benchmark_estimates_tokens_when_usage_omitted() -> None:
    transport = _build_healthy_transport(include_usage=False)
    with httpx.Client(transport=transport) as client:
        summary = run_benchmark(
            base_url="http://127.0.0.1:8000",
            api_key="yllm_test123_validsecret",
            runs=2,
            client=client,
        )

    assert summary.total_completion_tokens == 8  # 4 chunks * 2 runs
    assert all(s.estimated_tokens for s in summary.samples)


def test_run_benchmark_invalid_runs_raises() -> None:
    with pytest.raises(DiagnosticsError):
        run_benchmark(
            base_url="http://127.0.0.1:8000",
            api_key="yllm_test123_validsecret",
            runs=0,
        )


def test_cli_doctor_and_bench_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Typer CLI wiring for `doctor` and `bench` with --json and error exit codes."""
    fake_report = DoctorReport(
        base_url="http://127.0.0.1:8000",
        model_id="qwen2.5-3b-awq",
        checks=[
            CheckResult(name="Liveness (/health/live)", passed=True, latency_ms=4.2, detail="HTTP 200"),
            CheckResult(name="Tool-Call Compatibility", passed=True, latency_ms=120.5, detail="Verified"),
        ],
    )
    fake_summary = BenchmarkSummary(
        base_url="http://127.0.0.1:8000",
        model="qwen2.5-3b-awq",
        runs=2,
        avg_ttft_ms=85.0,
        min_ttft_ms=80.0,
        max_ttft_ms=90.0,
        avg_tps=42.5,
        min_tps=40.0,
        max_tps=45.0,
        avg_total_s=1.2,
        total_completion_tokens=100,
        samples=[
            BenchmarkSample(1, 80.0, 1.1, 20, 50, 45.0, "stop", False),
            BenchmarkSample(2, 90.0, 1.3, 20, 50, 40.0, "stop", False),
        ],
    )

    monkeypatch.setattr("yourllm.cli.main.run_doctor", lambda **_kwargs: fake_report)
    monkeypatch.setattr("yourllm.cli.main.run_benchmark", lambda **_kwargs: fake_summary)

    # 1. doctor --json
    doc_res = runner.invoke(
        app,
        ["doctor", "--url", "http://127.0.0.1:8000", "--api-key", "yllm_fake_secret", "--json"],
    )
    assert doc_res.exit_code == 0
    doc_data = json.loads(doc_res.stdout)
    assert doc_data["all_passed"] is True
    assert doc_data["model_id"] == "qwen2.5-3b-awq"

    # 2. bench --json
    bench_res = runner.invoke(
        app,
        ["bench", "--url", "http://127.0.0.1:8000", "--api-key", "yllm_fake_secret", "--runs", "2", "--json"],
    )
    assert bench_res.exit_code == 0
    bench_data = json.loads(bench_res.stdout)
    assert bench_data["avg_tps"] == 42.5
    assert bench_data["total_completion_tokens"] == 100

    # 3. bench without API key exits non-zero
    missing_key_res = runner.invoke(app, ["bench", "--url", "http://127.0.0.1:8000"])
    assert missing_key_res.exit_code == 1
