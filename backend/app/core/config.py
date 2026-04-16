"""Application settings from environment variables."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    """backend/app/core/config.py -> project root (CliniQ)."""
    return Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Override via environment or `.env` at repo root."""

    model_config = SettingsConfigDict(
        env_file=str(_default_project_root() / ".env"),
        env_file_encoding="utf-8",
        env_prefix="CLINIQ_",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_default_project_root)

    adapter_path: Optional[Path] = Field(
        default=None,
        description="Directory containing PEFT adapter (adapter_config.json).",
    )
    hf_token: Optional[str] = Field(default=None)

    device_map: str = Field(default="cuda:0")
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(default="bfloat16")
    load_in_4bit: bool = Field(default=False)

    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    cors_origins: str = Field(
        default="http://127.0.0.1:5173,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )

    skip_model_load: bool = Field(
        default=False,
        description="If true, lifespan skips GPU/PEFT load (for pytest / CI without CUDA).",
    )

    mock_generation: bool = Field(
        default=False,
        description="If true, /chat returns a fixed stub string (no torch); for UI/API integration only.",
    )

    @model_validator(mode="after")
    def fill_hf_token_from_env(self) -> "Settings":
        """Accept HF_TOKEN / HUGGING_FACE_HUB_TOKEN without CLINIQ_ prefix."""
        if self.hf_token is None:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if token:
                object.__setattr__(self, "hf_token", token)
        return self

    def resolved_adapter_path(self) -> Path:
        """Adapter directory; defaults to sft_adaptor under project root."""
        if self.adapter_path is not None:
            return Path(self.adapter_path).expanduser().resolve()
        return (self.project_root / "sft_adaptor").resolve()


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def clear_settings_cache() -> None:
    """Test hook to reload settings."""
    get_settings.cache_clear()
