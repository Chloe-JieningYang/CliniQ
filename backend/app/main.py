"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import health
from .api.v1_router import api_v1_router
from .core.config import Settings, get_settings
from .services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _ensure_hf_hub_env(settings: Settings) -> None:
    """huggingface_hub / LangChain often read HF_TOKEN from os.environ, not Pydantic."""
    token = settings.hf_token
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def _configure_app_package_logging() -> None:
    """Emit INFO for this app's loggers on stderr.

    Uvicorn is usually started as `uvicorn app.main:app` (cwd: backend), so loggers are
    `app.*`, not `backend.*`. Tests use `backend.*`. We attach to the top-level package
    derived from this module's name.
    """
    root_pkg = __name__.split(".", maxsplit=1)[0]
    pkg_log = logging.getLogger(root_pkg)
    pkg_log.setLevel(logging.INFO)
    if pkg_log.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    pkg_log.addHandler(handler)
    pkg_log.propagate = False


def _parse_cors_origins(raw: str) -> List[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load LLM once at startup; release reference on shutdown."""
    settings = get_settings()
    service = LLMService(settings)
    if settings.mock_generation:
        logger.info("CLINIQ_MOCK_GENERATION: stub chat replies (no PEFT / torch)")
    elif not settings.skip_model_load:
        service.load()
    else:
        logger.info("CLINIQ_SKIP_MODEL_LOAD: skipping PEFT / GPU model load")

    if settings.rag_enabled and not settings.mock_generation:
        service.load_rag()

    app.state.llm_service = service
    if service.is_loaded:
        if settings.mock_generation:
            logger.info("Chat API ready (mock generation)")
        else:
            logger.info("LLM ready at %s", settings.resolved_adapter_path())
    else:
        logger.error("LLM not loaded: %s", service.load_error)
    try:
        yield
    finally:
        app.state.llm_service = None


def create_app() -> FastAPI:
    _configure_app_package_logging()
    settings = get_settings()
    _ensure_hf_hub_env(settings)
    app = FastAPI(
        title="CliniQ API",
        description="Medical Q&A backend with PEFT adapters",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = _parse_cors_origins(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(api_v1_router)

    return app


app = create_app()
