"""Typed exception classes formatted for OpenAI-compatible error payloads."""

from __future__ import annotations

from typing import Any


class YourLLMError(Exception):
    """Base exception for all YourLLM errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "server_error",
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        """Convert the exception into an OpenAI-compatible error dictionary."""
        err: dict[str, Any] = {
            "message": self.message,
            "type": self.error_type,
            "code": self.code,
        }
        if self.hint:
            err["hint"] = self.hint
        return {"error": err}


class AuthenticationError(YourLLMError):
    """Raised when an API key is missing, malformed, or invalid (HTTP 401)."""

    def __init__(
        self,
        message: str = "Incorrect API key provided.",
        code: str = "invalid_api_key",
        hint: str | None = "Pass your API key in the 'Authorization: Bearer <key>' header.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_type="authentication_error",
            code=code,
            hint=hint,
        )


class PermissionError(YourLLMError):
    """Raised when an API key does not have permission for an endpoint (HTTP 403)."""

    def __init__(
        self,
        message: str = "You do not have permission to access this resource.",
        code: str = "insufficient_permissions",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=403,
            error_type="permission_error",
            code=code,
            hint=hint,
        )


class NotFoundError(YourLLMError):
    """Raised when a requested resource is not found (HTTP 404)."""

    def __init__(
        self,
        message: str = "The requested resource was not found.",
        code: str = "not_found",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=404,
            error_type="not_found_error",
            code=code,
            hint=hint,
        )


class PayloadTooLargeError(YourLLMError):
    """Raised when request payload exceeds size limit (HTTP 413)."""

    def __init__(
        self,
        message: str = "Request body exceeds maximum size limit.",
        code: str = "payload_too_large",
        hint: str | None = "Reduce request payload size or increase YLLM_MAX_REQUEST_BYTES.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=413,
            error_type="invalid_request_error",
            code=code,
            hint=hint,
        )


class InvalidRequestError(YourLLMError):
    """Raised when request parameters or schema are invalid (HTTP 400)."""

    def __init__(
        self,
        message: str,
        code: str = "invalid_request",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_type="invalid_request_error",
            code=code,
            hint=hint,
        )


class RateLimitError(YourLLMError):
    """Raised when concurrency or request cap is reached (HTTP 429)."""

    def __init__(
        self,
        message: str = "Server concurrency limit reached.",
        code: str = "concurrency_limit_exceeded",
        retry_after: int = 5,
        hint: str | None = "Please retry after a brief delay.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_error",
            code=code,
            hint=hint,
        )
        self.retry_after = retry_after


class ServiceUnavailableError(YourLLMError):
    """Raised when model or server is not ready (HTTP 503)."""

    def __init__(
        self,
        message: str = "Model is loading or server is not ready.",
        code: str = "server_not_ready",
        retry_after: int = 10,
        hint: str | None = "Poll /health/ready until status is READY.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_type="service_unavailable_error",
            code=code,
            hint=hint,
        )
        self.retry_after = retry_after


class BadGatewayError(YourLLMError):
    """Raised when upstream inference backend fails (HTTP 502)."""

    def __init__(
        self,
        message: str = "Upstream inference engine returned an error.",
        code: str = "upstream_error",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=502,
            error_type="server_error",
            code=code,
            hint=hint,
        )


class GatewayTimeoutError(YourLLMError):
    """Raised when upstream inference backend times out (HTTP 504)."""

    def __init__(
        self,
        message: str = "Upstream inference engine timed out.",
        code: str = "gateway_timeout",
        hint: str | None = "Consider increasing YLLM_REQUEST_TIMEOUT_S.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=504,
            error_type="server_error",
            code=code,
            hint=hint,
        )


class InternalServerError(YourLLMError):
    """Raised for unexpected internal server errors (HTTP 500)."""

    def __init__(
        self,
        message: str = "An internal error occurred.",
        code: str = "internal_error",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_type="server_error",
            code=code,
            hint=hint,
        )


# =====================================================================
# Hardware Detection Exceptions
# =====================================================================


class HardwareError(YourLLMError):
    """Base exception for hardware and GPU detection errors."""

    def __init__(
        self,
        message: str,
        code: str = "hardware_error",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_type="hardware_error",
            code=code,
            hint=hint,
        )


class NvidiaSmiMissingError(HardwareError):
    """Raised when nvidia-smi cannot be found in PATH."""

    def __init__(
        self,
        message: str = "nvidia-smi was not found in PATH.",
        hint: str | None = (
            "Ensure NVIDIA drivers are installed and nvidia-smi is available in PATH. "
            "If using Docker, ensure the container has GPU access enabled."
        ),
    ) -> None:
        super().__init__(message=message, code="nvidia_smi_missing", hint=hint)


class DriverError(HardwareError):
    """Raised when nvidia-smi fails due to driver or NVML issues."""

    def __init__(
        self,
        message: str = "NVIDIA driver or NVML communication failure.",
        hint: str | None = (
            "Ensure NVIDIA kernel module is loaded and host driver is functioning. "
            "Check dmesg and verify nvidia-smi executes successfully on the host."
        ),
    ) -> None:
        super().__init__(message=message, code="driver_error", hint=hint)


class NoDevicesError(HardwareError):
    """Raised when nvidia-smi returns no GPU devices."""

    def __init__(
        self,
        message: str = "No NVIDIA GPUs detected.",
        hint: str | None = (
            "No NVIDIA GPUs were detected. If running in Docker, ensure container was launched "
            "with '--gpus all' (or '--gpus device=...') and NVIDIA Container Toolkit is installed."
        ),
    ) -> None:
        super().__init__(message=message, code="no_devices", hint=hint)


class InsufficientGpuCountError(HardwareError):
    """Raised when fewer GPUs are detected than tensor_parallel_size."""

    def __init__(
        self,
        required: int,
        available: int,
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        msg = (
            message
            or f"Requested tensor_parallel_size={required} but only {available} GPU(s) detected."
        )
        h = hint or (
            f"Decrease YLLM_TENSOR_PARALLEL_SIZE to at most {available} or make more GPUs available."
        )
        super().__init__(message=msg, code="insufficient_gpu_count", hint=h)
        self.required = required
        self.available = available


class InsufficientVramError(HardwareError):
    """Raised when free VRAM is insufficient for the requested model."""

    def __init__(
        self,
        required_gb: float,
        available_gb: float,
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        msg = (
            message
            or f"Total free VRAM ({available_gb:.1f} GB) is below minimum requirement ({required_gb:.1f} GB)."
        )
        h = hint or (
            "Select a smaller model, enable quantization (AWQ/GPTQ/FP8), or reduce YLLM_MAX_MODEL_LEN."
        )
        super().__init__(message=msg, code="insufficient_vram", hint=h)
        self.required_gb = required_gb
        self.available_gb = available_gb


class LowDiskSpaceError(HardwareError):
    """Raised when free disk space on data directory is insufficient."""

    def __init__(
        self,
        path: str,
        required_gb: float,
        available_gb: float,
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        msg = (
            message
            or f"Insufficient disk space on {path}: {available_gb:.1f} GB free, {required_gb:.1f} GB required."
        )
        h = hint or (
            f"Free up disk space on {path} or mount a larger volume for model weights and cache."
        )
        super().__init__(message=msg, code="low_disk_space", hint=h)
        self.path = path
        self.required_gb = required_gb
        self.available_gb = available_gb


class UnsupportedArchitectureError(HardwareError):
    """Raised when GPU architecture compute capability is unsupported."""

    def __init__(
        self,
        device_name: str,
        compute_cap: str,
        min_supported: str = "7.0",
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        msg = (
            message
            or f"GPU '{device_name}' has compute capability {compute_cap}, below minimum required {min_supported}."
        )
        h = hint or (
            f"vLLM requires NVIDIA GPUs with compute capability >= {min_supported} "
            "(Volta V100, Turing T4/RTX 20xx, Ampere, Ada Lovelace, Hopper)."
        )
        super().__init__(message=msg, code="unsupported_architecture", hint=h)
        self.device_name = device_name
        self.compute_cap = compute_cap
        self.min_supported = min_supported


# =====================================================================
# Model Profiles & Manager Exceptions
# =====================================================================


class ProfileError(YourLLMError):
    """Base exception for model profile errors."""

    def __init__(
        self,
        message: str,
        code: str = "profile_error",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_type="profile_error",
            code=code,
            hint=hint,
        )


class ProfileNotFoundError(ProfileError):
    """Raised when a requested model profile cannot be found."""

    def __init__(
        self,
        profile_name: str,
        available_profiles: list[str] | None = None,
    ) -> None:
        avail_str = f" Available profiles: {', '.join(sorted(available_profiles))}." if available_profiles else ""
        msg = f"Model profile '{profile_name}' not found.{avail_str}"
        hint = "Verify YLLM_PROFILE setting or check available profiles in yourllm/models/profiles/."
        super().__init__(message=msg, code="profile_not_found", hint=hint)
        self.status_code = 404
        self.profile_name = profile_name


class InvalidProfileError(ProfileError):
    """Raised when a profile YAML schema is invalid."""

    def __init__(
        self,
        profile_name: str,
        details: str,
    ) -> None:
        msg = f"Invalid model profile '{profile_name}': {details}"
        super().__init__(message=msg, code="invalid_profile", hint="Fix the YAML profile definition.")
        self.profile_name = profile_name


class UnsupportedParserError(ProfileError):
    """Raised when a tool-call parser is not supported by pinned vLLM version."""

    def __init__(
        self,
        parser: str,
        allowed_parsers: list[str],
    ) -> None:
        msg = (
            f"Tool call parser '{parser}' is not supported by vLLM. "
            f"Allowed parsers: {', '.join(allowed_parsers)}."
        )
        hint = f"Choose one of the supported parsers: {', '.join(allowed_parsers)}."
        super().__init__(message=msg, code="unsupported_tool_call_parser", hint=hint)
        self.parser = parser
        self.allowed_parsers = allowed_parsers


class ModelManagerError(YourLLMError):
    """Base exception for model download and cache manager errors."""

    def __init__(
        self,
        message: str,
        code: str = "model_manager_error",
        status_code: int = 500,
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="model_manager_error",
            code=code,
            hint=hint,
        )


class ModelNotFoundError(ModelManagerError):
    """Raised when model repository cannot be found on Hugging Face."""

    def __init__(
        self,
        repo_id: str,
        details: str | None = None,
    ) -> None:
        msg = f"Model repository '{repo_id}' not found on Hugging Face."
        if details:
            msg += f" Details: {details}"
        hint = "Check the repository ID spelling on huggingface.co and verify network connectivity."
        super().__init__(message=msg, code="model_not_found", status_code=404, hint=hint)
        self.repo_id = repo_id


class GatedRepoError(ModelManagerError):
    """Raised when access to a Hugging Face model repository is restricted/gated."""

    def __init__(
        self,
        repo_id: str,
        details: str | None = None,
    ) -> None:
        msg = f"Access to Hugging Face repository '{repo_id}' is restricted or unauthorized."
        if details:
            msg += f" Details: {details}"
        hint = (
            f"1. Accept the license agreement on https://huggingface.co/{repo_id}.\n"
            "2. Generate a Hugging Face user access token at https://huggingface.co/settings/tokens.\n"
            "3. Provide the token via HF_TOKEN environment variable."
        )
        super().__init__(message=msg, code="gated_model_access_denied", status_code=403, hint=hint)
        self.repo_id = repo_id


# =====================================================================
# Lifecycle Exceptions
# =====================================================================


class LifecycleError(YourLLMError):
    """Base exception for lifecycle state machine errors."""

    def __init__(
        self,
        message: str,
        code: str = "lifecycle_error",
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_type="lifecycle_error",
            code=code,
            hint=hint,
        )


class InvalidStateTransitionError(LifecycleError):
    """Raised when attempting an illegal state transition."""

    def __init__(
        self,
        current_state: str,
        target_state: str,
        allowed_targets: set[str],
    ) -> None:
        allowed_str = ", ".join(sorted(allowed_targets)) if allowed_targets else "none"
        msg = (
            f"Invalid lifecycle transition from '{current_state}' to '{target_state}'. "
            f"Allowed next states: [{allowed_str}]."
        )
        hint = "Check the lifecycle state machine transition graph."
        super().__init__(message=msg, code="invalid_state_transition", hint=hint)
        self.current_state = current_state
        self.target_state = target_state
        self.allowed_targets = allowed_targets


class RestartLimitExceededError(LifecycleError):
    """Raised when inference crashes exceed the bounded recovery threshold."""

    def __init__(
        self,
        max_restarts: int,
        window_seconds: int,
        crash_count: int,
    ) -> None:
        msg = (
            f"Inference process crashed {crash_count} times within {window_seconds}s "
            f"(threshold: max {max_restarts}). Transitioning to FAILED."
        )
        hint = (
            "Inspect container and vLLM logs for the root cause of the crashes "
            "(e.g., CUDA out of memory, driver crash, or incompatible weights)."
        )
        super().__init__(message=msg, code="restart_limit_exceeded", hint=hint)
        self.max_restarts = max_restarts
        self.window_seconds = window_seconds
        self.crash_count = crash_count


# =====================================================================
# Inference & Process Supervision Exceptions (M4)
# =====================================================================


class InferenceProcessError(YourLLMError):
    """Base exception for inference engine child process failures."""

    def __init__(
        self,
        message: str,
        code: str = "inference_process_error",
        status_code: int = 503,
        hint: str | None = None,
        recent_logs: list[str] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="inference_process_error",
            code=code,
            hint=hint,
        )
        self.recent_logs = recent_logs or []


class InferenceStartupTimeoutError(InferenceProcessError):
    """Raised when vLLM fails to reach READY within the configured startup timeout."""

    def __init__(
        self,
        timeout_s: float,
        recent_logs: list[str] | None = None,
    ) -> None:
        msg = f"vLLM inference engine failed to reach READY state within {timeout_s:.0f} seconds."
        hint = (
            "Model loading may have timed out, or GPU initialization hung. "
            "Inspect trailing vLLM logs for details."
        )
        super().__init__(
            message=msg,
            code="inference_startup_timeout",
            hint=hint,
            recent_logs=recent_logs,
        )


class InferenceCrashError(InferenceProcessError):
    """Raised when the vLLM child process exits unexpectedly."""

    def __init__(
        self,
        exit_code: int | None,
        recent_logs: list[str] | None = None,
    ) -> None:
        msg = f"vLLM child process terminated unexpectedly with exit code {exit_code}."
        hint = (
            "vLLM may have encountered a CUDA out-of-memory error, invalid weight tensor, "
            "or segmentation fault. Check trailing logs."
        )
        super().__init__(
            message=msg,
            code="inference_process_crash",
            hint=hint,
            recent_logs=recent_logs,
        )


