"""Chat completion endpoint."""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.deps import get_llm_service
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    body: ChatRequest,
    llm: Annotated[LLMService, Depends(get_llm_service)],
) -> ChatResponse:
    """Run one turn of medical Q&A with role-conditioned system prompt."""
    if not llm.is_loaded:
        detail = llm.load_error or "Model is not loaded"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

    loop = asyncio.get_running_loop()

    def _generate() -> str:
        return llm.generate(body.role, body.message, body.context)

    try:
        answer = await loop.run_in_executor(None, _generate)
    except RuntimeError as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected generation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ChatResponse(answer=answer, role=body.role)
