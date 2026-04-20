"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, Request

from ..services.llm_service import LLMService


def get_llm_service(request: Request) -> LLMService:
    """Return the process-wide LLM service attached in app lifespan."""
    service = getattr(request.app.state, "llm_service", None)
    if service is None:
        raise RuntimeError("LLMService is not initialized")
    return service


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
