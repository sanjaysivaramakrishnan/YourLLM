"""Live server diagnostics (`doctor`) and streaming throughput benchmarking (`bench`)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BENCH_PROMPT = (
    "Write a concise Python function implementing binary search with type hints and a brief docstring."
)

TOOL_PROBE_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers and return their sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer."},
                    "b": {"type": "integer", "description": "Second integer."},
                },
                "required": ["a", "b"],
            },
        },
    }
]


class DiagnosticsError(Exception):
    """Raised when a benchmark or diagnostic precondition fails."""


def normalize_base_url(url: str) -> str:
    """Normalize a user-provided server URL by stripping trailing '/' and '/v1'."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned.rstrip("/")


@dataclass
class CheckResult:
    """Single diagnostic check outcome."""

    name: str
    passed: bool
    detail: str
    latency_ms: float | None = None
    warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "warning": self.warning,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
            "detail": self.detail,
        }


@dataclass
class DoctorReport:
    """Aggregated report from `yourllm-cli doctor`."""

    base_url: str
    model_id: str | None = None
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.warning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model_id": self.model_id,
            "all_passed": self.all_passed,
            "warning_count": self.warning_count,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class StreamProbeMetrics:
    """Parsed metrics and OpenAI protocol assertions from a single SSE stream."""

    ttft_ms: float
    total_s: float
    content_text: str
    content_chunks: int
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    saw_done: bool
    tool_calls_seen: bool
    tool_call_names: list[str]
    missing_tool_call_id: bool
    missing_tool_call_type: bool


@dataclass
class BenchmarkSample:
    """Measured timings and throughput for a single streaming benchmark run."""

    run_index: int
    ttft_ms: float
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int
    tokens_per_sec: float
    finish_reason: str | None
    estimated_tokens: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_s": round(self.total_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "finish_reason": self.finish_reason,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass
class BenchmarkSummary:
    """Aggregate throughput and latency summary across N benchmark runs."""

    base_url: str
    model: str
    runs: int
    avg_ttft_ms: float
    min_ttft_ms: float
    max_ttft_ms: float
    avg_tps: float
    min_tps: float
    max_tps: float
    avg_total_s: float
    total_completion_tokens: int
    samples: list[BenchmarkSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "runs": self.runs,
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "min_ttft_ms": round(self.min_ttft_ms, 2),
            "max_ttft_ms": round(self.max_ttft_ms, 2),
            "avg_tps": round(self.avg_tps, 2),
            "min_tps": round(self.min_tps, 2),
            "max_tps": round(self.max_tps, 2),
            "avg_total_s": round(self.avg_total_s, 3),
            "total_completion_tokens": self.total_completion_tokens,
            "samples": [s.to_dict() for s in self.samples],
        }


def _consume_sse_stream(response: httpx.Response, t_start: float) -> StreamProbeMetrics:
    """Consume an OpenAI SSE response stream and collect timing and compliance metrics."""
    ttft_ms: float | None = None
    content_parts: list[str] = []
    content_chunks = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    saw_done = False

    tool_calls_seen = False
    tool_call_names: list[str] = []
    seen_tc_indices: set[int] = set()
    missing_tool_call_id = False
    missing_tool_call_type = False

    for raw_line in response.iter_lines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if not isinstance(chunk, dict):
            continue

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if "prompt_tokens" in usage and usage["prompt_tokens"] is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if "completion_tokens" in usage and usage["completion_tokens"] is not None:
                completion_tokens = int(usage["completion_tokens"])

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - t_start) * 1000.0
                    content_parts.append(content)
                    content_chunks += 1

                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - t_start) * 1000.0
                    tool_calls_seen = True
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        idx = int(tc.get("index", 0))
                        if idx not in seen_tc_indices:
                            seen_tc_indices.add(idx)
                            if not tc.get("id"):
                                missing_tool_call_id = True
                            if tc.get("type") != "function":
                                missing_tool_call_type = True
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            fn_name = fn.get("name")
                            if isinstance(fn_name, str) and fn_name:
                                tool_call_names.append(fn_name)

            fr = choice.get("finish_reason")
            if isinstance(fr, str) and fr:
                finish_reason = fr

    total_s = max(time.monotonic() - t_start, 1e-6)
    final_ttft_ms = ttft_ms if ttft_ms is not None else total_s * 1000.0

    return StreamProbeMetrics(
        ttft_ms=final_ttft_ms,
        total_s=total_s,
        content_text="".join(content_parts),
        content_chunks=content_chunks,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        saw_done=saw_done,
        tool_calls_seen=tool_calls_seen,
        tool_call_names=tool_call_names,
        missing_tool_call_id=missing_tool_call_id,
        missing_tool_call_type=missing_tool_call_type,
    )


def _discover_model(
    http_client: httpx.Client,
    base_url: str,
    api_key: str,
    requested_model: str | None = None,
) -> tuple[str, list[str], float]:
    """Query `/v1/models` and return `(selected_model, available_models, latency_ms)`."""
    t0 = time.monotonic()
    resp = http_client.get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    latency_ms = (time.monotonic() - t0) * 1000.0

    if resp.status_code == 401:
        raise DiagnosticsError("Authentication rejected (HTTP 401). Check your API key.")
    if resp.status_code == 503:
        raise DiagnosticsError("Server is not ready (HTTP 503). Wait for model loading to finish.")
    if resp.status_code != 200:
        raise DiagnosticsError(f"GET /v1/models returned unexpected HTTP {resp.status_code}.")

    body = resp.json()
    raw_models = body.get("data", []) if isinstance(body, dict) else []
    available = [
        str(item["id"])
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not available:
        raise DiagnosticsError("No models returned by GET /v1/models.")

    if requested_model:
        if requested_model not in available:
            raise DiagnosticsError(
                f"Model '{requested_model}' not found. Available: {', '.join(available)}"
            )
        return requested_model, available, latency_ms

    return available[0], available, latency_ms


def run_doctor(
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout_s: float = 60.0,
    skip_tools: bool = False,
    client: httpx.Client | None = None,
) -> DoctorReport:
    """Run an end-to-end diagnostic suite against a live YourLLM gateway."""
    normalized_url = normalize_base_url(base_url)
    report = DoctorReport(base_url=normalized_url)

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_s)

    try:
        # 1. Liveness check
        t0 = time.monotonic()
        try:
            live_resp = http_client.get(f"{normalized_url}/health/live")
            live_ms = (time.monotonic() - t0) * 1000.0
        except httpx.RequestError as exc:
            report.checks.append(
                CheckResult(
                    name="Liveness (/health/live)",
                    passed=False,
                    detail=f"Cannot reach server at {normalized_url}: {type(exc).__name__}",
                )
            )
            return report

        if live_resp.status_code == 200:
            live_state = _extract_status(live_resp)
            report.checks.append(
                CheckResult(
                    name="Liveness (/health/live)",
                    passed=True,
                    latency_ms=live_ms,
                    detail=f"HTTP 200 (state: {live_state})",
                )
            )
        else:
            report.checks.append(
                CheckResult(
                    name="Liveness (/health/live)",
                    passed=False,
                    latency_ms=live_ms,
                    detail=f"Unexpected HTTP {live_resp.status_code}",
                )
            )
            return report

        # 2. Readiness check
        t0 = time.monotonic()
        ready_resp = http_client.get(f"{normalized_url}/health/ready")
        ready_ms = (time.monotonic() - t0) * 1000.0
        ready_state = _extract_status(ready_resp)
        if ready_resp.status_code == 200 and ready_state == "READY":
            report.checks.append(
                CheckResult(
                    name="Readiness (/health/ready)",
                    passed=True,
                    latency_ms=ready_ms,
                    detail="Server is READY for inference",
                )
            )
        else:
            report.checks.append(
                CheckResult(
                    name="Readiness (/health/ready)",
                    passed=False,
                    latency_ms=ready_ms,
                    detail=f"Not ready (HTTP {ready_resp.status_code}, state: {ready_state})",
                )
            )
            return report

        # 3. Unauthenticated rejection check (security invariant)
        t0 = time.monotonic()
        unauth_resp = http_client.get(f"{normalized_url}/v1/models")
        unauth_ms = (time.monotonic() - t0) * 1000.0
        if unauth_resp.status_code == 401:
            report.checks.append(
                CheckResult(
                    name="Auth Enforcement",
                    passed=True,
                    latency_ms=unauth_ms,
                    detail="Unauthenticated /v1/models properly rejected (HTTP 401)",
                )
            )
        else:
            report.checks.append(
                CheckResult(
                    name="Auth Enforcement",
                    passed=False,
                    latency_ms=unauth_ms,
                    detail=f"Expected HTTP 401 without API key, got HTTP {unauth_resp.status_code}",
                )
            )

        # 4. Authenticated model discovery
        if not api_key:
            report.checks.append(
                CheckResult(
                    name="Model Discovery (/v1/models)",
                    passed=False,
                    detail="API key not provided. Pass --api-key or set YLLM_API_KEY.",
                )
            )
            return report

        try:
            target_model, available_models, models_ms = _discover_model(
                http_client=http_client,
                base_url=normalized_url,
                api_key=api_key,
                requested_model=model,
            )
            report.model_id = target_model
            report.checks.append(
                CheckResult(
                    name="Model Discovery (/v1/models)",
                    passed=True,
                    latency_ms=models_ms,
                    detail=f"Authenticated; active model: '{target_model}' ({len(available_models)} available)",
                )
            )
        except DiagnosticsError as exc:
            report.checks.append(
                CheckResult(
                    name="Model Discovery (/v1/models)",
                    passed=False,
                    detail=str(exc),
                )
            )
            return report

        auth_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 5. Streaming chat completion probe
        stream_payload: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": "Reply with exactly the word OK."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t0 = time.monotonic()
        with http_client.stream(
            "POST",
            f"{normalized_url}/v1/chat/completions",
            headers=auth_headers,
            json=stream_payload,
        ) as stream_resp:
            if stream_resp.status_code != 200:
                report.checks.append(
                    CheckResult(
                        name="Streaming Completion",
                        passed=False,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        detail=f"HTTP {stream_resp.status_code} from /v1/chat/completions",
                    )
                )
                return report

            metrics = _consume_sse_stream(stream_resp, t0)

        stream_issues: list[str] = []
        if not metrics.saw_done:
            stream_issues.append("missing [DONE] marker")
        if metrics.finish_reason is None:
            stream_issues.append("missing finish_reason")
        if not metrics.content_text.strip() and not metrics.tool_calls_seen:
            stream_issues.append("empty response")

        if stream_issues:
            report.checks.append(
                CheckResult(
                    name="Streaming Completion",
                    passed=False,
                    latency_ms=metrics.total_s * 1000.0,
                    detail=f"SSE spec issue: {', '.join(stream_issues)}",
                )
            )
        else:
            report.checks.append(
                CheckResult(
                    name="Streaming Completion",
                    passed=True,
                    latency_ms=metrics.total_s * 1000.0,
                    detail=(
                        f"TTFT {metrics.ttft_ms:.0f} ms, finish_reason='{metrics.finish_reason}', "
                        "[DONE] received"
                    ),
                )
            )

        # 6. Coding-agent streaming tool-call compatibility check
        if not skip_tools:
            tool_payload: dict[str, Any] = {
                "model": target_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call the add_numbers tool with a=2 and b=3 right now.",
                    }
                ],
                "tools": TOOL_PROBE_SCHEMA,
                "tool_choice": "auto",
                "max_tokens": 96,
                "temperature": 0.0,
                "stream": True,
            }
            t0 = time.monotonic()
            with http_client.stream(
                "POST",
                f"{normalized_url}/v1/chat/completions",
                headers=auth_headers,
                json=tool_payload,
            ) as tool_resp:
                if tool_resp.status_code != 200:
                    report.checks.append(
                        CheckResult(
                            name="Tool-Call Compatibility",
                            passed=False,
                            latency_ms=(time.monotonic() - t0) * 1000.0,
                            detail=f"HTTP {tool_resp.status_code} on tool-call request",
                        )
                    )
                else:
                    tool_metrics = _consume_sse_stream(tool_resp, t0)
                    if tool_metrics.tool_calls_seen:
                        tc_issues: list[str] = []
                        if tool_metrics.missing_tool_call_id:
                            tc_issues.append("initial delta missing tool_call id")
                        if tool_metrics.missing_tool_call_type:
                            tc_issues.append("initial delta missing type='function'")
                        if tool_metrics.finish_reason != "tool_calls":
                            tc_issues.append(
                                f"expected finish_reason='tool_calls', got '{tool_metrics.finish_reason}'"
                            )
                        if not tool_metrics.saw_done:
                            tc_issues.append("missing [DONE] marker")

                        if tc_issues:
                            report.checks.append(
                                CheckResult(
                                    name="Tool-Call Compatibility",
                                    passed=False,
                                    latency_ms=tool_metrics.total_s * 1000.0,
                                    detail=f"OpenAI/Cline mismatch: {', '.join(tc_issues)}",
                                )
                            )
                        else:
                            names = ", ".join(tool_metrics.tool_call_names) or "function"
                            report.checks.append(
                                CheckResult(
                                    name="Tool-Call Compatibility",
                                    passed=True,
                                    latency_ms=tool_metrics.total_s * 1000.0,
                                    detail=(
                                        f"Verified ({names}): id/type present, "
                                        "finish_reason='tool_calls', [DONE] ok"
                                    ),
                                )
                            )
                    else:
                        # Stream worked, but the model replied in plain text rather than emitting a tool call
                        report.checks.append(
                            CheckResult(
                                name="Tool-Call Compatibility",
                                passed=True,
                                warning=True,
                                latency_ms=tool_metrics.total_s * 1000.0,
                                detail=(
                                    "Stream ok, but model replied in plain text instead of calling the tool "
                                    "(check YLLM_TOOL_CALL_PARSER or model size)."
                                ),
                            )
                        )

        return report
    finally:
        if owns_client:
            http_client.close()


def run_benchmark(
    base_url: str,
    api_key: str,
    model: str | None = None,
    runs: int = 3,
    max_tokens: int = 128,
    prompt: str = DEFAULT_BENCH_PROMPT,
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
    on_sample: Callable[[BenchmarkSample], None] | None = None,
) -> BenchmarkSummary:
    """Measure streaming TTFT and generation throughput (tokens/sec) over N runs."""
    if runs < 1:
        raise DiagnosticsError("'runs' must be at least 1.")
    if max_tokens < 1:
        raise DiagnosticsError("'max_tokens' must be at least 1.")
    if not api_key:
        raise DiagnosticsError("API key is required. Pass --api-key or set YLLM_API_KEY.")

    normalized_url = normalize_base_url(base_url)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_s)

    try:
        try:
            target_model, _, _ = _discover_model(
                http_client=http_client,
                base_url=normalized_url,
                api_key=api_key,
                requested_model=model,
            )
        except httpx.RequestError as exc:
            raise DiagnosticsError(
                f"Cannot reach server at {normalized_url}: {type(exc).__name__}"
            ) from exc

        auth_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        samples: list[BenchmarkSample] = []
        for idx in range(1, runs + 1):
            t0 = time.monotonic()
            try:
                with http_client.stream(
                    "POST",
                    f"{normalized_url}/v1/chat/completions",
                    headers=auth_headers,
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        raise DiagnosticsError(
                            f"Benchmark run {idx} failed with HTTP {resp.status_code}."
                        )
                    probe = _consume_sse_stream(resp, t0)
            except httpx.RequestError as exc:
                raise DiagnosticsError(
                    f"Benchmark run {idx} network error: {type(exc).__name__}"
                ) from exc

            if probe.completion_tokens is not None and probe.completion_tokens > 0:
                comp_tokens = probe.completion_tokens
                estimated = False
            else:
                comp_tokens = max(probe.content_chunks, 0)
                estimated = True

            tps = comp_tokens / probe.total_s if comp_tokens > 0 else 0.0
            sample = BenchmarkSample(
                run_index=idx,
                ttft_ms=probe.ttft_ms,
                total_s=probe.total_s,
                prompt_tokens=probe.prompt_tokens,
                completion_tokens=comp_tokens,
                tokens_per_sec=tps,
                finish_reason=probe.finish_reason,
                estimated_tokens=estimated,
            )
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)

        ttfts = [s.ttft_ms for s in samples]
        tps_vals = [s.tokens_per_sec for s in samples]
        totals = [s.total_s for s in samples]

        return BenchmarkSummary(
            base_url=normalized_url,
            model=target_model,
            runs=runs,
            avg_ttft_ms=sum(ttfts) / len(ttfts),
            min_ttft_ms=min(ttfts),
            max_ttft_ms=max(ttfts),
            avg_tps=sum(tps_vals) / len(tps_vals),
            min_tps=min(tps_vals),
            max_tps=max(tps_vals),
            avg_total_s=sum(totals) / len(totals),
            total_completion_tokens=sum(s.completion_tokens for s in samples),
            samples=samples,
        )
    finally:
        if owns_client:
            http_client.close()


def _extract_status(resp: httpx.Response) -> str:
    """Safely extract the 'status' string from a health endpoint JSON body."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            status = data.get("status")
            if isinstance(status, str):
                return status
    except Exception:
        pass
    return "UNKNOWN"
