"""Subprocess manager for supervising the internal vLLM inference engine."""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from yourllm.core.exceptions import InferenceProcessError

logger = logging.getLogger(__name__)
vllm_logger = logging.getLogger("yourllm.vllm")

# Pattern for redacting sensitive secrets from raw subprocess streams
SECRET_PATTERN = re.compile(r"(hf_[a-zA-Z0-9_]{16,}|yllm_[a-zA-Z0-9_]{20,})")


def sanitize_vllm_log(line: str) -> str:
    """Mask any Hugging Face or YourLLM API keys that may appear in logs."""
    return SECRET_PATTERN.sub("[REDACTED]", line)


class VLLMProcessManager:
    """Async supervisor managing the vLLM subprocess lifecycle, process group, and logs."""

    def __init__(
        self,
        max_log_lines: int = 100,
        on_unexpected_exit: Callable[[int | None], Any] | None = None,
    ) -> None:
        self.max_log_lines = max_log_lines
        self.on_unexpected_exit = on_unexpected_exit

        self._process: asyncio.subprocess.Process | None = None
        self._is_stopping = False
        self._log_ring_buffer: collections.deque[str] = collections.deque(maxlen=max_log_lines)
        self._reader_tasks: list[asyncio.Task[None]] = []
        self._watcher_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        """True if the subprocess exists and has not terminated."""
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        """Process ID of the managed child process."""
        return self._process.pid if self._process else None

    @property
    def returncode(self) -> int | None:
        """Exit code of the process if terminated, or None if running."""
        return self._process.returncode if self._process else None

    @property
    def recent_logs(self) -> list[str]:
        """Snapshot of trailing log lines captured from stdout and stderr."""
        return list(self._log_ring_buffer)

    async def start(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        """Spawn the vLLM child process in an isolated process group.

        Args:
            cmd: Command vector (must never be a shell string).
            env: Process environment variables.
        """
        if self.is_running:
            raise InferenceProcessError("vLLM process is already running.")

        self._is_stopping = False
        self._log_ring_buffer.clear()

        kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env,
        }

        # On POSIX, use start_new_session=True to create a new process group.
        # This allows sending SIGTERM/SIGKILL to the entire group (worker processes).
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        logger.info("Spawning vLLM child process: %s", " ".join(cmd[:3]) + " ...")
        try:
            self._process = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        except Exception as exc:
            raise InferenceProcessError(f"Failed to spawn vLLM child process: {exc}") from exc

        logger.info("vLLM child process started with PID %s.", self._process.pid)

        # Start background stream reader tasks
        if self._process.stdout:
            self._reader_tasks.append(
                asyncio.create_task(self._read_stream(self._process.stdout, stream_name="stdout"))
            )
        if self._process.stderr:
            self._reader_tasks.append(
                asyncio.create_task(self._read_stream(self._process.stderr, stream_name="stderr"))
            )

        # Start watcher task to detect unexpected process termination
        self._watcher_task = asyncio.create_task(self._watch_process())

    async def _read_stream(self, stream: asyncio.StreamReader, stream_name: str) -> None:
        """Asynchronously read lines from subprocess stdout/stderr and store in ring buffer."""
        try:
            while not stream.at_eof():
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace").rstrip("\r\n")
                if not line:
                    continue
                safe_line = sanitize_vllm_log(line)
                self._log_ring_buffer.append(f"[{stream_name}] {safe_line}")
                if stream_name == "stderr":
                    vllm_logger.warning("%s", safe_line)
                else:
                    vllm_logger.info("%s", safe_line)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Exception reading vLLM %s stream: %s", stream_name, exc)

    async def _watch_process(self) -> None:
        """Monitor process exit and trigger unexpected exit callback if not shutting down."""
        if not self._process:
            return
        try:
            exit_code = await self._process.wait()
            if not self._is_stopping:
                logger.error(
                    "vLLM child process (PID %s) exited unexpectedly with code %s.",
                    self._process.pid,
                    exit_code,
                )
                if self.on_unexpected_exit:
                    try:
                        res = self.on_unexpected_exit(exit_code)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as exc:
                        logger.error("Error in on_unexpected_exit callback: %s", exc, exc_info=True)
        except asyncio.CancelledError:
            pass

    async def stop(self, grace_period_s: float = 15.0) -> int | None:
        """Gracefully terminate vLLM child process and its process group (SIGTERM -> grace -> SIGKILL).

        Args:
            grace_period_s: Seconds to wait for SIGTERM before sending SIGKILL.

        Returns:
            Subprocess returncode.
        """
        if self._process is None or self._process.returncode is not None:
            return self._process.returncode if self._process else None

        self._is_stopping = True
        pid = self._process.pid
        logger.info("Stopping vLLM process (PID %s) with SIGTERM (grace period: %.1fs)...", pid, grace_period_s)

        # Send SIGTERM to process group if supported, otherwise directly to process
        self._signal_process_group(signal.SIGTERM)

        try:
            # Wait for graceful exit
            await asyncio.wait_for(self._process.wait(), timeout=grace_period_s)
            logger.info("vLLM process (PID %s) exited gracefully with code %s.", pid, self._process.returncode)
        except TimeoutError:
            logger.warning(
                "vLLM process (PID %s) did not exit within %.1fs. Sending SIGKILL to process group.",
                pid,
                grace_period_s,
            )
            sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            self._signal_process_group(sigkill if sys.platform != "win32" else signal.SIGTERM)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
                logger.info("vLLM process (PID %s) terminated via SIGKILL.", pid)
            except TimeoutError:
                logger.error("vLLM process (PID %s) failed to terminate even after SIGKILL.", pid)

        # Cancel stream readers and watcher
        for task in self._reader_tasks:
            if not task.done():
                task.cancel()
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()

        return self._process.returncode

    def _signal_process_group(self, sig: int) -> None:
        """Deliver signal to the process group on POSIX or terminate on Windows."""
        if not self._process or self._process.returncode is not None:
            return

        pid = self._process.pid
        if sys.platform != "win32" and hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
            except Exception as exc:
                logger.debug("os.killpg failed for PID %s: %s; falling back to direct signal", pid, exc)

        try:
            sigkill = getattr(signal, "SIGKILL", None)
            if sigkill is not None and sig == sigkill:
                self._process.kill()
            else:
                self._process.terminate()
        except ProcessLookupError:
            pass
        except Exception as exc:
            logger.debug("Direct signal failed for PID %s: %s", pid, exc)
