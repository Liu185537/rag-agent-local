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
    """创建隔离的 TestClient。

    设计目的：
    - 每次测试使用独立 SQLite/Chroma 路径，避免数据互相污染；
    - 强制关闭 LLM（RAG_AGENT_LLM_PROVIDER=none），让测试稳定且可离线执行。
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 为本次测试运行生成独立目录。
    run_root = project_root / "data" / "test_runs" / f"run_{uuid.uuid4().hex[:8]}"
    sqlite_path = run_root / "test_rag_agent.db"
    chroma_path = run_root / "chroma"
    run_root.mkdir(parents=True, exist_ok=True)

    # 用环境变量覆盖应用配置。
    monkeypatch.setenv("RAG_AGENT_SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("RAG_AGENT_CHROMA_PATH", str(chroma_path))
    monkeypatch.setenv("RAG_AGENT_LLM_PROVIDER", "none")
    monkeypatch.setenv("RAG_AGENT_ENV", "test")

    from app.core import config as config_module

    # 清理配置缓存并重载主模块，确保新环境变量生效。
    config_module.get_settings.cache_clear()
    import app.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client

    # 测试结束后清理临时目录。
    shutil.rmtree(run_root, ignore_errors=True)
