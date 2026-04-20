"""Application settings from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field
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
    hf_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "CLINIQ_HF_TOKEN",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        ),
        description="Hugging Face token; use HF_TOKEN in .env or CLINIQ_HF_TOKEN.",
    )

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

    rag_enabled: bool = Field(
        default=False,
        description="If true, load FAISS index at startup and prepend retrieved context before generation.",
    )
    rag_index_path: Optional[Path] = Field(
        default=None,
        description="Directory with FAISS index; default rag/faiss_index under project root.",
    )
    rag_top_k: int = Field(default=1, ge=1, le=32)
    rag_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Same embedding model as rag/build_vector.py.",
    )

    def resolved_adapter_path(self) -> Path:
        """Adapter directory; defaults to sft_adaptor under project root."""
        if self.adapter_path is not None:
            return Path(self.adapter_path).expanduser().resolve()
        return (self.project_root / "sft_adaptor").resolve()

    def resolved_rag_index_path(self) -> Path:
        """FAISS directory produced by rag/build_vector.py (default rag/faiss_index)."""
        if self.rag_index_path is not None:
            return Path(self.rag_index_path).expanduser().resolve()
        return (self.project_root / "rag" / "faiss_index").resolve()


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def clear_settings_cache() -> None:
    """Test hook to reload settings."""
    get_settings.cache_clear()
