"""Chat API schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """User chat request."""

    role: Literal["doctor", "patient"] = Field(
        ...,
        description="Audience: doctor (clinical depth) or patient (plain language).",
    )
    message: str = Field(..., min_length=1, max_length=8000)
    context: Optional[str] = Field(
        default=None,
        max_length=16000,
        description="Optional clinical context or background.",
    )


class ChatResponse(BaseModel):
    """Model reply."""

    answer: str
    role: Literal["doctor", "patient"]
