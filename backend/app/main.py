"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import health
from .api.v1_router import api_v1_router
from .core.config import get_settings
from .services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _parse_cors_origins(raw: str) -> List[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load LLM once at startup; release reference on shutdown."""
    settings = get_settings()
    service = LLMService(settings)
    if not settings.skip_model_load:
        service.load()
    else:
        logger.info("CLINIQ_SKIP_MODEL_LOAD: skipping PEFT / GPU model load")
    app.state.llm_service = service
    if service.is_loaded:
        logger.info("LLM ready at %s", settings.resolved_adapter_path())
    else:
        logger.error("LLM not loaded: %s", service.load_error)
    try:
        yield
    finally:
        app.state.llm_service = None


def create_app() -> FastAPI:
    settings = get_settings()
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
