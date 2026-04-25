"""Chat works with CLINIQ_MOCK_GENERATION (no torch)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.config import clear_settings_cache
from backend.app.main import create_app


def test_mock_generation_chat_returns_stub(monkeypatch) -> None:
    monkeypatch.setenv("CLINIQ_MOCK_GENERATION", "true")
    monkeypatch.setenv("CLINIQ_SKIP_MODEL_LOAD", "true")
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as client:
        model = client.get("/api/v1/model").json()
        assert model.get("mock_generation") is True
        assert model.get("real_weights_loaded") is False

        response = client.post(
            "/api/v1/chat",
            json={
                "role": "patient",
                "message": "Demo question",
                "context": "Demo context",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "patient"
    assert "Mock mode" in body["answer"]
    assert "Demo question" in body["answer"]
