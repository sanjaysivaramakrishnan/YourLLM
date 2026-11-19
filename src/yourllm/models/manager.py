"""Hugging Face model download and local cache manager."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from yourllm.core.exceptions import (
    GatedRepoError,
    LowDiskSpaceError,
    ModelManagerError,
    ModelNotFoundError,
)

try:
    from huggingface_hub import model_info, snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError as HfGatedRepoError,
    )
    from huggingface_hub.errors import (
        HfHubHTTPError,
        RepositoryNotFoundError,
    )
except ImportError:
    model_info = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]
    HfGatedRepoError = Exception  # type: ignore[misc,assignment]
    RepositoryNotFoundError = Exception  # type: ignore[misc,assignment]
    HfHubHTTPError = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model weight downloads, local caching, and pre-flight validation."""

    def __init__(
        self,
        cache_dir: Path | str = "/data/models/huggingface",
        hf_token: SecretStr | str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._hf_token = hf_token

    def _get_token_str(self) -> str | None:
        """Safely retrieve plaintext token string without logging or exposing it."""
        if self._hf_token is None:
            return None
        if isinstance(self._hf_token, SecretStr):
            return self._hf_token.get_secret_value()
        return str(self._hf_token)

    def check_disk_space(
        self,
        required_bytes: int,
        target_path: Path | None = None,
    ) -> None:
        """Validate sufficient disk space before starting a download.

        Args:
            required_bytes: Required space in bytes.
            target_path: Path to inspect (defaults to cache_dir).

        Raises:
            LowDiskSpaceError: If available space is less than required.
        """
        path = target_path or self.cache_dir
        # Walk up to existing ancestor if path does not exist yet
        check_dir = path
        while not check_dir.exists() and check_dir.parent != check_dir:
            check_dir = check_dir.parent

        try:
            usage = shutil.disk_usage(str(check_dir))
            free_bytes = usage.free
        except Exception:
            # Fallback if disk usage query fails
            return

        if free_bytes < required_bytes:
            raise LowDiskSpaceError(
                path=str(path),
                required_gb=round(required_bytes / (1024.0**3), 2),
                available_gb=round(free_bytes / (1024.0**3), 2),
            )

    def get_repo_size_bytes(self, repo_id: str) -> int | None:
        """Query Hugging Face Hub for total model file sizes.

        Returns total bytes if available, or None if undetermined.
        """
        if model_info is None:
            return None

        token_str = self._get_token_str()
        try:
            info = model_info(repo_id=repo_id, token=token_str)
            if hasattr(info, "siblings") and info.siblings:
                total_bytes = 0
                has_sizes = False
                for s in info.siblings:
                    size = getattr(s, "size", None)
                    if size is not None:
                        total_bytes += int(size)
                        has_sizes = True
                if has_sizes and total_bytes > 0:
                    return total_bytes
        except Exception:
            return None
        return None

    def find_cached_snapshot(self, repo_id: str) -> Path | None:
        """Check if a complete snapshot for repo_id exists in local cache without network calls."""
        safe_name = "models--" + repo_id.replace("/", "--")
        snapshots_dir = self.cache_dir / safe_name / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if snapshots:
                # Check if refs/main exists to pick current commit
                refs_main = self.cache_dir / safe_name / "refs" / "main"
                if refs_main.is_file():
                    try:
                        commit_hash = refs_main.read_text(encoding="utf-8").strip()
                        commit_dir = snapshots_dir / commit_hash
                        if commit_dir.is_dir():
                            return commit_dir.resolve()
                    except Exception:
                        pass
                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                return latest_snapshot.resolve()
        return None

    def prepare_model(
        self,
        model_id_or_path: str,
        progress_callback: Callable[[float], Any] | None = None,
    ) -> Path:
        """Verify, download, and cache model weights into the target cache directory.

        Args:
            model_id_or_path: Hugging Face repo ID (e.g. meta-llama/Llama-3.1-8B-Instruct)
                              or an existing local directory path.
            progress_callback: Optional progress indicator callback.

        Returns:
            Path to the local directory containing model weights.

        Raises:
            ModelNotFoundError: If the repo does not exist.
            GatedRepoError: If access is forbidden or gated without valid token.
            LowDiskSpaceError: If disk space is insufficient.
            ModelManagerError: On network or download failure.
        """
        local_candidate = Path(model_id_or_path)
        if local_candidate.is_dir():
            logger.info("Using local model directory: %s", local_candidate)
            return local_candidate.resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Zero-network fast path: check if model snapshot already exists in local cache
        cached_snapshot = self.find_cached_snapshot(model_id_or_path)
        if cached_snapshot is not None:
            logger.info(
                "Model '%s' verified in local cache: %s (skipping all network checks).",
                model_id_or_path,
                cached_snapshot,
            )
            if progress_callback:
                progress_callback(1.0)
            return cached_snapshot

        token_str = self._get_token_str()

        # 2. Pre-check disk space using repository metadata if available
        repo_size = self.get_repo_size_bytes(model_id_or_path)
        if repo_size is not None:
            # Add 10% safety margin for temporary files / unpacked layers
            required_with_margin = int(repo_size * 1.1)
            self.check_disk_space(required_with_margin, target_path=self.cache_dir)

        # 3. Model not in cache — download snapshot from Hugging Face Hub
        logger.info(
            "Model '%s' not found in local cache. Initiating one-time download to %s",
            model_id_or_path,
            self.cache_dir,
        )

        if snapshot_download is None:
            raise ModelManagerError(
                "huggingface_hub library is not installed or available."
            )

        try:
            local_path = snapshot_download(
                repo_id=model_id_or_path,
                cache_dir=str(self.cache_dir),
                token=token_str,
                force_download=False,
            )
            if progress_callback:
                progress_callback(1.0)
            return Path(local_path)

        except RepositoryNotFoundError as exc:
            raise ModelNotFoundError(repo_id=model_id_or_path, details=str(exc)) from exc

        except HfGatedRepoError as exc:
            raise GatedRepoError(repo_id=model_id_or_path, details=str(exc)) from exc

        except HfHubHTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (401, 403):
                raise GatedRepoError(repo_id=model_id_or_path, details=str(exc)) from exc
            if status_code == 404:
                raise ModelNotFoundError(repo_id=model_id_or_path, details=str(exc)) from exc
            raise ModelManagerError(
                f"Failed to download model '{model_id_or_path}' due to HTTP error: {exc}"
            ) from exc

        except LowDiskSpaceError:
            raise

        except Exception as exc:
            # Check message text for 401/403/gated signals
            err_msg = str(exc)
            if "401" in err_msg or "403" in err_msg or "gated" in err_msg.lower():
                raise GatedRepoError(repo_id=model_id_or_path, details=err_msg) from exc
            raise ModelManagerError(
                f"Failed to download model '{model_id_or_path}': {err_msg}"
            ) from exc
