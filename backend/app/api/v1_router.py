"""All routes under /api/v1."""

from typing import Annotated

from fastapi import APIRouter, Depends

from ..core.deps import get_llm_service
from ..services.llm_service import LLMService
from .routes import chat

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(chat.router)


@api_v1_router.get("/model", tags=["health"])
def model_info(llm: Annotated[LLMService, Depends(get_llm_service)]) -> dict:
    """Summarize adapter path, base model id, and load state (no secrets)."""
    return llm.model_card()
