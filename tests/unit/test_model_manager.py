"""Unit tests for model download and cache manager with mocked huggingface_hub."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from yourllm.core.exceptions import (
    GatedRepoError,
    LowDiskSpaceError,
    ModelManagerError,
    ModelNotFoundError,
)
from yourllm.models.manager import ModelManager


def test_prepare_local_directory(tmp_path: Path) -> None:
    local_model = tmp_path / "my_local_model"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}", encoding="utf-8")

    manager = ModelManager(cache_dir=tmp_path / "cache")
    result = manager.prepare_model(str(local_model))
    assert result == local_model.resolve()


@patch("yourllm.models.manager.snapshot_download")
@patch("yourllm.models.manager.model_info")
def test_prepare_download_success(
    mock_model_info: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
) -> None:
    mock_model_info.return_value = MagicMock(siblings=[])
    expected_path = tmp_path / "downloaded_model"
    expected_path.mkdir()
    mock_snapshot.return_value = str(expected_path)

    progress_calls: list[float] = []

    def on_progress(p: float) -> None:
        progress_calls.append(p)

    manager = ModelManager(
        cache_dir=tmp_path / "hf_cache",
        hf_token=SecretStr("hf_secret_12345"),
    )
    result = manager.prepare_model("Qwen/Qwen2.5-7B-Instruct", progress_callback=on_progress)

    assert result == expected_path
    assert len(progress_calls) == 1
    assert progress_calls[0] == 1.0

    mock_snapshot.assert_called_once()
    called_kwargs = mock_snapshot.call_args.kwargs
    assert called_kwargs.get("repo_id") == "Qwen/Qwen2.5-7B-Instruct"
    assert called_kwargs.get("force_download") is False
    assert called_kwargs.get("token") == "hf_secret_12345"


@patch("yourllm.models.manager.snapshot_download")
@patch("yourllm.models.manager.model_info")
def test_prepare_gated_repo(
    mock_model_info: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
) -> None:
    mock_model_info.return_value = MagicMock(siblings=[])
    mock_snapshot.side_effect = Exception("403 Client Error: Repository is gated. Cannot access.")

    manager = ModelManager(cache_dir=tmp_path / "hf_cache")
    with pytest.raises(GatedRepoError) as exc_info:
        manager.prepare_model("meta-llama/Llama-3.1-8B-Instruct")

    assert "meta-llama/Llama-3.1-8B-Instruct" in exc_info.value.message
    assert "HF_TOKEN" in str(exc_info.value.hint)


@patch("yourllm.models.manager.snapshot_download")
@patch("yourllm.models.manager.model_info")
def test_prepare_model_not_found(
    mock_model_info: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
) -> None:
    mock_model_info.return_value = MagicMock(siblings=[])
    from huggingface_hub.errors import RepositoryNotFoundError

    mock_response = MagicMock(status_code=404)
    mock_snapshot.side_effect = RepositoryNotFoundError(
        "404 Repository not found",
        response=mock_response,
    )

    manager = ModelManager(cache_dir=tmp_path / "hf_cache")
    with pytest.raises(ModelNotFoundError) as exc_info:
        manager.prepare_model("nonexistent/unknown-repo")

    assert "nonexistent/unknown-repo" in exc_info.value.message


@patch("yourllm.models.manager.shutil.disk_usage")
@patch("yourllm.models.manager.model_info")
def test_prepare_low_disk_space(
    mock_model_info: MagicMock,
    mock_disk_usage: MagicMock,
    tmp_path: Path,
) -> None:
    # Model size 20 GB
    mock_model_info.return_value = MagicMock(
        siblings=[MagicMock(size=20 * 1024 * 1024 * 1024)]
    )
    # Disk free only 5 GB
    mock_disk_usage.return_value = MagicMock(free=5 * 1024 * 1024 * 1024)

    manager = ModelManager(cache_dir=tmp_path / "hf_cache")
    with pytest.raises(LowDiskSpaceError) as exc_info:
        manager.prepare_model("large/model-weights")

    assert "Insufficient disk space" in exc_info.value.message


@patch("yourllm.models.manager.snapshot_download")
@patch("yourllm.models.manager.model_info")
def test_token_is_never_logged(
    mock_model_info: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_token = "hf_SUPER_SECRET_TOKEN_VALUE_XYZ"
    mock_model_info.return_value = MagicMock(siblings=[])
    mock_snapshot.return_value = str(tmp_path)

    manager = ModelManager(
        cache_dir=tmp_path / "hf_cache",
        hf_token=SecretStr(secret_token),
    )
    with caplog.at_level("DEBUG"):
        manager.prepare_model("test/repo")

    for record in caplog.records:
        assert secret_token not in record.message


@patch("yourllm.models.manager.snapshot_download")
@patch("yourllm.models.manager.model_info")
def test_network_failure_wrapped(
    mock_model_info: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
) -> None:
    mock_model_info.return_value = MagicMock(siblings=[])
    mock_snapshot.side_effect = RuntimeError("Connection timed out after 30s")

    manager = ModelManager(cache_dir=tmp_path / "hf_cache")
    with pytest.raises(ModelManagerError) as exc_info:
        manager.prepare_model("test/repo")
    assert "Connection timed out" in exc_info.value.message


@patch("yourllm.models.manager.snapshot_download")
@patch("yourllm.models.manager.model_info")
def test_prepare_cached_snapshot_zero_network(
    mock_model_info: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "hf_cache"
    snapshot_dir = cache_dir / "models--Qwen--Qwen2.5-3B-Instruct-AWQ" / "snapshots" / "abc123commit"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    manager = ModelManager(cache_dir=cache_dir)
    result = manager.prepare_model("Qwen/Qwen2.5-3B-Instruct-AWQ")

    assert result == snapshot_dir.resolve()
    # Confirm zero network calls were attempted
    mock_snapshot.assert_not_called()
    mock_model_info.assert_not_called()

