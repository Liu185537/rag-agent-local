from __future__ import annotations

import io

import pytest


def test_health(client):
    """健康检查接口应返回服务状态和测试环境标识。"""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["env"] == "test"


def test_ingest_retrieve_chat_history_metrics(client):
    """覆盖核心链路：入库 -> 检索 -> 对话 -> 历史 -> 指标。"""

    # 1) 入库：写入一条销售话术样例文档。
    ingest_payload = {
        "namespace": "demo_test",
        "docs": [
            {
                "doc_id": "doc_sales",
                "text": (
                    "When price objection appears, use empathy first. "
                    "Then explain measurable value and propose a trial class."
                ),
                "source": "unit_test",
                "metadata": {"topic": "sales"},
            }
        ],
    }
    ingest_resp = client.post("/api/v1/knowledge/ingest", json=ingest_payload)
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["ingested_docs"] == 1
    assert ingest_data["ingested_chunks"] >= 1

    # 2) 检索：应至少召回一个分块，且包含重排分字段。
    retrieve_payload = {"query": "How to handle price objection?", "namespace": "demo_test", "top_k": 3}
    retrieve_resp = client.post("/api/v1/retrieve", json=retrieve_payload)
    assert retrieve_resp.status_code == 200
    retrieve_data = retrieve_resp.json()
    assert len(retrieve_data["chunks"]) > 0
    assert "rerank_score" in retrieve_data["chunks"][0]

    # 3) 对话：应返回会话 ID 与非空答案。
    chat_payload = {"message": "How should I answer price objection?", "namespace": "demo_test"}
    chat_resp = client.post("/api/v1/chat", json=chat_payload)
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["session_id"]
    assert chat_data["answer"]

    # 4) 历史：至少包含用户和助手两条消息。
    history_resp = client.get(f"/api/v1/sessions/{chat_data['session_id']}/history")
    assert history_resp.status_code == 200
    history_data = history_resp.json()
    assert len(history_data["messages"]) >= 2

    # 5) 指标：以上调用应累积多次请求。
    metrics_resp = client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert metrics_data["total_requests"] >= 4


def test_upload_ingest(client):
    """文件上传入库接口：验证 markdown 文件可被解析并切块。"""
    content = b"# Notes\n\nThis is a follow-up cadence note with 48 hours and next step."
    files = {"file": ("upload_notes.md", content, "text/markdown")}
    data = {"namespace": "upload_ns", "doc_id": "upload_doc_1"}
    resp = client.post("/api/v1/knowledge/upload", files=files, data=data)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["namespace"] == "upload_ns"
    assert payload["ingested_chunks"] >= 1


def test_upload_ingest_xlsx(client):
    """文件上传入库接口：验证 xlsx 文件可被解析并切块。"""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "leads"
    ws.append(["name", "status", "next_step"])
    ws.append(["Alice", "busy", "follow-up in 48h"])
    ws.append(["Bob", "interested", "book trial class"])

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    files = {
        "file": (
            "upload_leads.xlsx",
            buffer.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    data = {"namespace": "upload_xlsx_ns", "doc_id": "upload_xlsx_doc_1"}
    resp = client.post("/api/v1/knowledge/upload", files=files, data=data)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["namespace"] == "upload_xlsx_ns"
    assert payload["ingested_chunks"] >= 1


def test_agent_config_crud(client):
    """Agent 配置 CRUD：默认读取 -> 保存 -> 再读 -> 删除。"""

    # 默认状态应走内置配置（source=default）。
    get_default = client.get("/api/v1/agent-config", params={"namespace": "demo_cfg"})
    assert get_default.status_code == 200
    data_default = get_default.json()
    assert data_default["source"] == "default"
    assert data_default["config"]["name"]

    # 写入自定义配置。
    payload = {
        "namespace": "demo_cfg",
        "name": "Sales Coach",
        "description": "Helps with enrollment messages",
        "instructions": "Give concise and structured sales responses.",
    }
    save_resp = client.put("/api/v1/agent-config", json=payload)
    assert save_resp.status_code == 200
    save_data = save_resp.json()
    assert save_data["source"] == "storage"
    assert save_data["config"]["name"] == "Sales Coach"

    # 再次读取应命中持久化配置。
    get_saved = client.get("/api/v1/agent-config", params={"namespace": "demo_cfg"})
    assert get_saved.status_code == 200
    get_saved_data = get_saved.json()
    assert get_saved_data["source"] == "storage"
    assert get_saved_data["config"]["instructions"].startswith("Give concise")

    # 删除后不再保留该配置。
    delete_resp = client.delete("/api/v1/agent-config", params={"namespace": "demo_cfg"})
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True


def test_demo_summary(client):
    """演示汇总接口：应返回 namespace、统计信息与指标快照。"""
    ns = "summary_ns"
    ingest_payload = {
        "namespace": ns,
        "docs": [
            {
                "doc_id": "doc_summary",
                "text": "A short summary doc for demo summary endpoint testing.",
                "source": "unit_test",
                "metadata": {"topic": "summary"},
            }
        ],
    }
    ingest_resp = client.post("/api/v1/knowledge/ingest", json=ingest_payload)
    assert ingest_resp.status_code == 200

    summary_resp = client.get("/api/v1/demo/summary", params={"namespace": ns})
    assert summary_resp.status_code == 200
    data = summary_resp.json()
    assert data["namespace"] == ns
    assert "stats" in data
    assert data["stats"]["doc_count"] >= 1
    assert "metrics" in data
