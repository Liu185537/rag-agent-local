from __future__ import annotations

from typing import Any

from app.core.database import Database
from app.rag.retriever import HybridRetriever, RetrievedChunk


class ToolRegistry:
    def __init__(self, db: Database, retriever: HybridRetriever):
        self.db = db
        self.retriever = retriever

    def retrieve_knowledge(self, query: str, namespace: str, top_k: int) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query=query, namespace=namespace, top_k=top_k)

    def get_profile(self, session_id: str) -> dict[str, str]:
        return self.db.get_profile(session_id)

    def update_profile(self, session_id: str, updates: list[dict[str, Any]]) -> None:
        for item in updates:
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                self.db.upsert_profile(session_id=session_id, key=key, value=value)

