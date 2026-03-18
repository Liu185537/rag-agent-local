from __future__ import annotations

from typing import Any

from app.core.database import Database
from app.rag.retriever import HybridRetriever, RetrievedChunk


class ToolRegistry:
    """Agent 可调用工具集合。

    这里把“检索”和“画像读写”统一封装，便于后续扩展新工具。
    """

    def __init__(self, db: Database, retriever: HybridRetriever):
        self.db = db
        self.retriever = retriever

    def retrieve_knowledge(self, query: str, namespace: str, top_k: int) -> list[RetrievedChunk]:
        """知识检索工具：返回候选分块。"""
        return self.retriever.retrieve(query=query, namespace=namespace, top_k=top_k)

    def get_profile(self, session_id: str) -> dict[str, str]:
        """画像读取工具。"""
        return self.db.get_profile(session_id)

    def update_profile(self, session_id: str, updates: list[dict[str, Any]]) -> None:
        """画像更新工具：过滤空 key/value 后逐条写入。"""
        for item in updates:
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                self.db.upsert_profile(session_id=session_id, key=key, value=value)

