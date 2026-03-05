from __future__ import annotations

import importlib
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    run_root = project_root / "data" / "test_runs" / f"run_{uuid.uuid4().hex[:8]}"
    sqlite_path = run_root / "test_rag_agent.db"
    chroma_path = run_root / "chroma"
    run_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RAG_AGENT_SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("RAG_AGENT_CHROMA_PATH", str(chroma_path))
    monkeypatch.setenv("RAG_AGENT_LLM_PROVIDER", "none")
    monkeypatch.setenv("RAG_AGENT_ENV", "test")

    from app.core import config as config_module

    config_module.get_settings.cache_clear()
    import app.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client
    shutil.rmtree(run_root, ignore_errors=True)
