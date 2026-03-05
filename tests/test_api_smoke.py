from __future__ import annotations


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["env"] == "test"


def test_ingest_retrieve_chat_history_metrics(client):
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

    retrieve_payload = {"query": "How to handle price objection?", "namespace": "demo_test", "top_k": 3}
    retrieve_resp = client.post("/api/v1/retrieve", json=retrieve_payload)
    assert retrieve_resp.status_code == 200
    retrieve_data = retrieve_resp.json()
    assert len(retrieve_data["chunks"]) > 0
    assert "rerank_score" in retrieve_data["chunks"][0]

    chat_payload = {"message": "How should I answer price objection?", "namespace": "demo_test"}
    chat_resp = client.post("/api/v1/chat", json=chat_payload)
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["session_id"]
    assert chat_data["answer"]

    history_resp = client.get(f"/api/v1/sessions/{chat_data['session_id']}/history")
    assert history_resp.status_code == 200
    history_data = history_resp.json()
    assert len(history_data["messages"]) >= 2

    metrics_resp = client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert metrics_data["total_requests"] >= 4


def test_upload_ingest(client):
    content = b"# Notes\n\nThis is a follow-up cadence note with 48 hours and next step."
    files = {"file": ("upload_notes.md", content, "text/markdown")}
    data = {"namespace": "upload_ns", "doc_id": "upload_doc_1"}
    resp = client.post("/api/v1/knowledge/upload", files=files, data=data)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["namespace"] == "upload_ns"
    assert payload["ingested_chunks"] >= 1


def test_agent_config_crud(client):
    get_default = client.get("/api/v1/agent-config", params={"namespace": "demo_cfg"})
    assert get_default.status_code == 200
    data_default = get_default.json()
    assert data_default["source"] == "default"
    assert data_default["config"]["name"]

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

    get_saved = client.get("/api/v1/agent-config", params={"namespace": "demo_cfg"})
    assert get_saved.status_code == 200
    get_saved_data = get_saved.json()
    assert get_saved_data["source"] == "storage"
    assert get_saved_data["config"]["instructions"].startswith("Give concise")

    delete_resp = client.delete("/api/v1/agent-config", params={"namespace": "demo_cfg"})
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True
