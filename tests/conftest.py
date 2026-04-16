"""Pytest: force env before `backend.app.main` import (that module calls `create_app()` at import)."""

from __future__ import annotations

import os

import pytest

# Do not use setdefault: a developer shell or CI may export SKIP=false / MOCK=true.
os.environ["CLINIQ_SKIP_MODEL_LOAD"] = "true"
os.environ["CLINIQ_MOCK_GENERATION"] = "false"

from backend.app.core.config import clear_settings_cache

clear_settings_cache()


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    """Avoid stale `get_settings()` lru_cache between tests (e.g. after CLINIQ_MOCK_GENERATION)."""
    clear_settings_cache()
    yield
    clear_settings_cache()
