"""OpenAI-compatible models catalog routes (/v1/models)."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from yourllm.api.schemas import ModelListResponse, ModelObject
from yourllm.auth.dependencies import require_api_key
from yourllm.core.config import Settings
from yourllm.core.exceptions import NotFoundError, ServiceUnavailableError
from yourllm.core.lifecycle import LifecycleManager
from yourllm.storage.repositories import APIKeyRecord

router = APIRouter(prefix="/v1/models", tags=["models"])


def _check_ready(request: Request) -> None:
    lifecycle: LifecycleManager | None = getattr(request.app.state, "lifecycle", None)
    if lifecycle and not lifecycle.is_ready:
        raise ServiceUnavailableError(
            message="Model is loading or server is not ready.",
            code="server_not_ready",
            retry_after=10,
            hint="Poll /health/ready until status is READY.",
        )


@router.get("", response_model=ModelListResponse)
@router.get("/", response_model=ModelListResponse, include_in_schema=False)
async def list_models(
    request: Request,
    _key_record: Annotated[APIKeyRecord, Depends(require_api_key)],
) -> ModelListResponse:
    """List the served model in OpenAI list format."""
    _check_ready(request)

    settings: Settings = request.app.state.settings
    served_name = settings.served_model_name or settings.model

    model_obj = ModelObject(
        id=served_name,
        created=int(time.time()),
        owned_by="yourllm",
    )
    return ModelListResponse(data=[model_obj])


@router.get("/{model_id:path}", response_model=ModelObject)
async def get_model(
    model_id: str,
    request: Request,
    _key_record: Annotated[APIKeyRecord, Depends(require_api_key)],
) -> ModelObject:
    """Retrieve details for a specific model ID."""
    _check_ready(request)

    settings: Settings = request.app.state.settings
    served_name = settings.served_model_name or settings.model
    canonical_model = settings.model

    # Check if requested model matches served model name or HF repo name
    if model_id not in {served_name, canonical_model}:
        raise NotFoundError(
            message=f"The model '{model_id}' does not exist.",
            code="model_not_found",
            hint=f"Available model: '{served_name}'.",
        )

    return ModelObject(
        id=model_id,
        created=int(time.time()),
        owned_by="yourllm",
    )
