"""Model profile schema, registry, and configuration resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from yourllm.core.config import Settings
from yourllm.core.exceptions import (
    InvalidProfileError,
    ProfileNotFoundError,
    UnsupportedParserError,
)

logger = logging.getLogger(__name__)

# Pinned parser list verified for vLLM v0.7.3 (DECISIONS.md §1 & §5)
PINNED_VLLM_PARSERS = [
    "hermes",
    "llama3_json",
    "mistral",
    "granite",
    "pythonic",
    "openai",
]

DEFAULT_PROFILES_DIR = Path(__file__).parent / "profiles"


class ModelProfile(BaseModel):
    """Schema for a curated model profile."""

    name: str
    hf_repo: str
    status: Literal["untested", "expected", "tested"] = "untested"
    quantization: str | None = None
    tool_call_parser: str | None = None
    chat_template: str | None = None
    recommended_max_model_len: int | None = None
    min_total_vram_gb: int | None = None
    extra_vllm_args: list[str] = Field(default_factory=list)
    notes: str = ""
    tested_with: list[str] = Field(default_factory=list)

    @field_validator("tool_call_parser")
    @classmethod
    def validate_parser(cls, v: str | None) -> str | None:
        if v is not None and v not in PINNED_VLLM_PARSERS:
            raise UnsupportedParserError(v, PINNED_VLLM_PARSERS)
        return v


@dataclass(frozen=True)
class ResolvedModelConfig:
    """Fully merged and resolved model configuration ready for vLLM startup."""

    model: str
    served_model_name: str
    profile_name: str | None
    profile_status: str | None
    quantization: str | None
    tool_call_parser: str | None
    chat_template: str | None
    max_model_len: int | None
    min_total_vram_gb: int | None
    extra_vllm_args: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ProfileRegistry:
    """In-memory registry of loaded model profiles."""

    def __init__(self, profiles_dir: Path | None = None) -> None:
        self.profiles_dir = profiles_dir or DEFAULT_PROFILES_DIR
        self._profiles: dict[str, ModelProfile] = {}
        self.load_profiles()

    def load_profiles(self) -> None:
        """Scan directory and load all YAML profile definitions."""
        self._profiles.clear()
        if not self.profiles_dir.exists():
            return

        for file_path in self.profiles_dir.glob("*.yaml"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                profile = ModelProfile(**data)
                self._profiles[profile.name] = profile
            except UnsupportedParserError:
                raise
            except Exception as exc:
                raise InvalidProfileError(file_path.stem, str(exc)) from exc

    def get(self, name: str) -> ModelProfile | None:
        """Retrieve profile by unique profile name."""
        return self._profiles.get(name)

    def find_by_repo(self, hf_repo: str) -> ModelProfile | None:
        """Find profile matching Hugging Face repository ID."""
        for profile in self._profiles.values():
            if profile.hf_repo.lower() == hf_repo.lower() or profile.name.lower() == hf_repo.lower():
                return profile
        return None

    def list_all(self) -> list[ModelProfile]:
        """List all loaded model profiles."""
        return list(self._profiles.values())

    def list_names(self) -> list[str]:
        """List all registered profile names."""
        return sorted(self._profiles.keys())


def resolve_model_config(
    settings: Settings,
    registry: ProfileRegistry | None = None,
) -> ResolvedModelConfig:
    """Resolve runtime settings against profile registry and apply override precedence.

    Rules per CLAUDE.md §5 & §6:
    1. If YLLM_PROFILE is explicitly specified and != 'auto', look up by name. If not found, fail.
    2. If YLLM_PROFILE == 'auto', match by HF repo ID. If no match, proceed unprofiled.
    3. Env vars override profile values.
    4. Overriding a profile value logs a clear warning.
    5. Final tool_call_parser must be in the pinned vLLM parser list.

    Args:
        settings: Application Settings.
        registry: ProfileRegistry instance (defaults to loading standard profiles).

    Returns:
        ResolvedModelConfig.
    """
    reg = registry or ProfileRegistry()
    profile: ModelProfile | None = None
    warnings: list[str] = []

    profile_req = settings.profile.strip()
    if profile_req and profile_req.lower() != "auto":
        profile = reg.get(profile_req)
        if not profile:
            raise ProfileNotFoundError(profile_req, available_profiles=reg.list_names())
    else:
        # Auto-match by model repository ID
        profile = reg.find_by_repo(settings.model)

    profile_name = profile.name if profile else None
    profile_status = profile.status if profile else None

    # Base values from profile (if matched)
    base_quant = profile.quantization if profile else None
    base_parser = profile.tool_call_parser if profile else None
    base_template = profile.chat_template if profile else None
    base_max_len = profile.recommended_max_model_len if profile else None
    base_min_vram = profile.min_total_vram_gb if profile else None
    extra_args = list(profile.extra_vllm_args) if profile else []

    # Heuristic fallback for unprofiled popular models to guarantee tool calling support
    if profile is None:
        repo_lower = settings.model.lower()
        if "qwen" in repo_lower:
            base_parser = "hermes"
            base_template = "/opt/yourllm/templates/tool_chat_template_hermes.jinja"
            logger.info("Auto-inferred tool_call_parser='hermes' for Qwen model '%s'.", settings.model)
        elif "llama-3" in repo_lower or "llama3" in repo_lower:
            base_parser = "llama3_json"
            base_template = "/opt/yourllm/templates/tool_chat_template_llama3_json.jinja"
            logger.info("Auto-inferred tool_call_parser='llama3_json' for Llama-3 model '%s'.", settings.model)

    # Apply environment overrides
    quantization = base_quant
    if settings.quantization is not None:
        if profile and profile.quantization != settings.quantization:
            warn = (
                f"Overriding profile '{profile.name}' quantization: "
                f"'{profile.quantization}' -> '{settings.quantization}'."
            )
            warnings.append(warn)
            logger.warning(warn)
        quantization = settings.quantization

    tool_call_parser = base_parser
    if settings.tool_call_parser is not None:
        if profile and profile.tool_call_parser != settings.tool_call_parser:
            warn = (
                f"Overriding profile '{profile.name}' tool_call_parser: "
                f"'{profile.tool_call_parser}' -> '{settings.tool_call_parser}'."
            )
            warnings.append(warn)
            logger.warning(warn)
        tool_call_parser = settings.tool_call_parser

    chat_template = base_template
    if settings.chat_template is not None:
        if profile and profile.chat_template != settings.chat_template:
            warn = (
                f"Overriding profile '{profile.name}' chat_template: "
                f"'{profile.chat_template}' -> '{settings.chat_template}'."
            )
            warnings.append(warn)
            logger.warning(warn)
        chat_template = settings.chat_template

    max_model_len = settings.max_model_len or base_max_len

    # Validate final tool_call_parser against pinned vLLM parser list
    if tool_call_parser is not None and tool_call_parser not in PINNED_VLLM_PARSERS:
        raise UnsupportedParserError(tool_call_parser, PINNED_VLLM_PARSERS)

    served_model_name = settings.served_model_name or settings.model

    return ResolvedModelConfig(
        model=settings.model,
        served_model_name=served_model_name,
        profile_name=profile_name,
        profile_status=profile_status,
        quantization=quantization,
        tool_call_parser=tool_call_parser,
        chat_template=chat_template,
        max_model_len=max_model_len,
        min_total_vram_gb=base_min_vram,
        extra_vllm_args=extra_args,
        warnings=warnings,
    )
