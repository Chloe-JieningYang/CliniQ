"""Health endpoint (no LLM)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_ok() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
