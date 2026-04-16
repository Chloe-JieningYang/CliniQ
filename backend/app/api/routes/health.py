"""Liveness endpoint (outside /api/v1)."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Process is up (does not guarantee GPU model is loaded)."""
    return {"status": "ok"}
