"""Pytest fixtures: force skip real model load before any import of `backend.app.main`."""

from __future__ import annotations

import os

# Must run before `backend.app.main` is imported (that module calls `create_app()` at import time).
os.environ.setdefault("CLINIQ_SKIP_MODEL_LOAD", "true")

from backend.app.core.config import clear_settings_cache

clear_settings_cache()
