"""API tests for authentication enforcement, admin separation, and health probes."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from tests.fixtures.fake_vllm import FakeVLLMServer
from yourllm.auth.service import create_key, revoke_key
from yourllm.core.config import Settings
from yourllm.core.lifecycle import LifecycleManager, LifecycleState
from yourllm.inference.client import InferenceClient
from yourllm.main import create_app
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner


@pytest.fixture
def test_setup(tmp_path: Path) -> dict:
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()

    # Create a valid test key
    key_record, raw_key = create_key(db, name="test-key")

    # Create a revoked test key
    revoked_rec, revoked_raw = create_key(db, name="revoked-key")
    revoke_key(db, revoked_rec.key_id)

    fake_server = FakeVLLMServer()
    settings = Settings(
        YLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct",
        YLLM_DATA_DIR=tmp_path,
        YLLM_ADMIN_TOKEN=SecretStr("admin_super_secret_token"),
    )
    lifecycle = LifecycleManager(initial_state=LifecycleState.READY)
    inference_client = InferenceClient(
        base_url="http://fake-vllm",
        transport=httpx.ASGITransport(app=fake_server.app),
    )

    app = create_app(
        settings=settings,
        db=db,
        lifecycle=lifecycle,
        inference_client=inference_client,
    )

    return {
        "app": app,
        "valid_key": raw_key,
        "revoked_key": revoked_raw,
        "db": db,
        "lifecycle": lifecycle,
        "fake_server": fake_server,
        "admin_token": "admin_super_secret_token",
    }


@pytest.mark.asyncio
async def test_health_live_endpoint(test_setup: dict) -> None:
    app = test_setup["app"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "READY"}
        # Must leak nothing else
        assert "model" not in data
        assert "hardware" not in data
        assert resp.headers.get("X-Request-ID") is not None


@pytest.mark.asyncio
async def test_health_ready_endpoint(test_setup: dict) -> None:
    app = test_setup["app"]
    lifecycle: LifecycleManager = test_setup["lifecycle"]
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Ready state -> 200
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "READY"}

        # Non-ready state -> 503 with Retry-After
        lifecycle.transition_to(LifecycleState.STOPPING)
        resp_not_ready = await client.get("/health/ready")
        assert resp_not_ready.status_code == 503
        assert resp_not_ready.headers.get("Retry-After") == "10"
        assert resp_not_ready.json() == {"status": "STOPPING"}


@pytest.mark.asyncio
async def test_v1_auth_enforcement(test_setup: dict) -> None:
    app = test_setup["app"]
    valid_key = test_setup["valid_key"]
    revoked_key = test_setup["revoked_key"]
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Missing Authorization header -> 401
        resp = await client.get("/v1/models")
        assert resp.status_code == 401
        err = resp.json()["error"]
        assert err["type"] == "authentication_error"
        assert err["code"] == "missing_api_key"

        # 2. Malformed Bearer header -> 401
        resp_malformed = await client.get("/v1/models", headers={"Authorization": "Token 12345"})
        assert resp_malformed.status_code == 401

        # 3. Invalid key string -> 401
        resp_invalid = await client.get(
            "/v1/models", headers={"Authorization": "Bearer yllm_0000000000_invalidsecretkey123456789012"}
        )
        assert resp_invalid.status_code == 401

        # 4. Revoked key -> 401
        resp_revoked = await client.get(
            "/v1/models", headers={"Authorization": f"Bearer {revoked_key}"}
        )
        assert resp_revoked.status_code == 401
        assert "revoked" in resp_revoked.json()["error"]["message"].lower()

        # 5. Valid key -> 200
        resp_valid = await client.get(
            "/v1/models", headers={"Authorization": f"Bearer {valid_key}"}
        )
        assert resp_valid.status_code == 200


@pytest.mark.asyncio
async def test_admin_token_separation(test_setup: dict) -> None:
    app = test_setup["app"]
    valid_inference_key = test_setup["valid_key"]
    admin_token = test_setup["admin_token"]
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Inference key on admin route -> 401
        resp_inf = await client.get(
            "/admin/v1/config",
            headers={"Authorization": f"Bearer {valid_inference_key}"},
        )
        assert resp_inf.status_code == 401

        # 2. Missing admin token -> 401
        resp_none = await client.get("/admin/v1/config")
        assert resp_none.status_code == 401

        # 3. Invalid admin token -> 401
        resp_bad = await client.get(
            "/admin/v1/config",
            headers={"Authorization": "Bearer wrong_admin_token"},
        )
        assert resp_bad.status_code == 401

        # 4. Valid admin token on /admin/v1/config -> 200 with redacted config
        resp_ok = await client.get(
            "/admin/v1/config",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp_ok.status_code == 200
        cfg = resp_ok.json()
        assert cfg["YLLM_ADMIN_TOKEN"] == "[REDACTED]"
        assert "meta-llama/Llama-3.1-8B-Instruct" in cfg["YLLM_MODEL"]

        # 5. Valid admin token on /admin/v1/status -> 200
        resp_status = await client.get(
            "/admin/v1/status",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp_status.status_code == 200
        status_data = resp_status.json()
        assert status_data["lifecycle"]["state"] == "READY"


@pytest.mark.asyncio
async def test_admin_returns_404_when_unset(tmp_path: Path) -> None:
    # When YLLM_ADMIN_TOKEN is unset, admin routes return 404 by design
    settings_no_admin = Settings(
        YLLM_MODEL="test-model",
        YLLM_DATA_DIR=tmp_path,
        YLLM_ADMIN_TOKEN=None,
    )
    app = create_app(settings=settings_no_admin)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/v1/status", headers={"Authorization": "Bearer any_token"})
        assert resp.status_code == 404
