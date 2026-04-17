"""Model metadata and chat API with mocked LLM (no GPU)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi.testclient import TestClient

from backend.app.core.deps import get_llm_service
from backend.app.main import create_app


class _MockLLMService:
    """Minimal stand-in for `LLMService` used by routes."""

    def __init__(self) -> None:
        self.is_loaded: bool = True
        self.load_error: Optional[str] = None

    def generate(self, role: str, message: str, context: Optional[str]) -> str:
        ctx = f"|ctx={context!s}" if context else ""
        return f"[mock]{role}:{message[:48]}{ctx}"

    def model_card(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "adapter_path": "/mock/adapter",
            "device_map": "cpu",
            "load_in_4bit": False,
            "torch_dtype": "bfloat16",
            "base_model_name_or_path": "mock/Meta-Llama",
        }


def test_model_info_without_override_reports_not_loaded() -> None:
    """Real `LLMService` in app.state when skip_model_load: no weights on GPU."""
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body.get("mock_generation") is False
    assert body.get("real_weights_loaded") is False
    assert body.get("loaded") is False
    assert "adapter_path" in body


def test_chat_returns_503_when_model_not_loaded_and_no_override() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "role": "patient",
                "message": "What is hypertension?",
                "context": None,
            },
        )
    assert response.status_code == 503
    assert "detail" in response.json()


def test_chat_success_with_mock_llm() -> None:
    app = create_app()
    app.dependency_overrides[get_llm_service] = lambda: _MockLLMService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                json={
                    "role": "doctor",
                    "message": "Differential for acute chest pain?",
                    "context": "Male, 55, smoker",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "doctor"
    assert data["answer"].startswith("[mock]doctor:")
    assert "ctx=" in data["answer"]


def test_chat_validation_empty_message() -> None:
    app = create_app()
    app.dependency_overrides[get_llm_service] = lambda: _MockLLMService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                json={"role": "patient", "message": ""},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
