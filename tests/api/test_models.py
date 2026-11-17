"""API tests for the /v1/models routes."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from yourllm.auth.service import create_key
from yourllm.core.config import Settings
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.main import create_app
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner


@pytest.fixture
def app_and_key(tmp_path: Path) -> tuple[httpx.AsyncClient, str, LifecycleManager]:
    db = Database(tmp_path / "test.db")
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    rec, raw_key = create_key(db, name="models-test-key")

    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_SERVED_MODEL_NAME="llama-3.1-8b",
        YLLM_DATA_DIR=tmp_path,
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.READY)
    app = create_app(settings=settings, db=db, lifecycle=lifecycle)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return client, raw_key, lifecycle


@pytest.mark.asyncio
async def test_list_models_success(app_and_key: tuple[httpx.AsyncClient, str, LifecycleManager]) -> None:
    client, key, _ = app_and_key
    resp = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 1
    model = data["data"][0]
    assert model["id"] == "llama-3.1-8b"
    assert model["object"] == "model"
    assert model["owned_by"] == "yourllm"


@pytest.mark.asyncio
async def test_get_model_by_id_success(app_and_key: tuple[httpx.AsyncClient, str, LifecycleManager]) -> None:
    client, key, _ = app_and_key
    # Lookup by served model name
    resp = await client.get("/v1/models/llama-3.1-8b", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    assert resp.json()["id"] == "llama-3.1-8b"

    # Lookup by repo ID
    resp_repo = await client.get(
        "/v1/models/meta-llama/Llama-3.1-8B-Instruct",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp_repo.status_code == 200
    assert resp_repo.json()["id"] == "meta-llama/Llama-3.1-8B-Instruct"


@pytest.mark.asyncio
async def test_get_unknown_model_returns_404(
    app_and_key: tuple[httpx.AsyncClient, str, LifecycleManager],
) -> None:
    client, key, _ = app_and_key
    resp = await client.get("/v1/models/gpt-4o", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "model_not_found"
    assert "gpt-4o" in err["message"]


@pytest.mark.asyncio
async def test_models_not_ready_returns_503(
    app_and_key: tuple[httpx.AsyncClient, str, LifecycleManager],
) -> None:
    client, key, lifecycle = app_and_key
    lifecycle.transition_to(LifecycleState.STOPPING)

    resp = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"
    assert resp.json()["error"]["code"] == "server_not_ready"
