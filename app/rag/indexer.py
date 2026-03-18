from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import Settings
from app.rag.embedding import EmbeddingService


def _normalize_namespace(namespace: str) -> str:
    """将 namespace 规范化为 Chroma collection 名称。"""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", namespace.strip().lower())
    if not cleaned:
        cleaned = "default"
    return cleaned[:63]


@dataclass
class VectorHit:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class ChromaIndexer:
    """Chroma 向量索引封装。"""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_path))

    def _get_collection(self, namespace: str) -> Collection:
        """按 namespace 获取（或创建）集合。"""
        return self._client.get_or_create_collection(name=_normalize_namespace(namespace))

    def upsert(self, namespace: str, records: list[dict[str, Any]]) -> None:
        """批量写入/更新分块向量。"""
        if not records:
            return
        collection = self._get_collection(namespace)
        collection.upsert(
            ids=[item["chunk_id"] for item in records],
            documents=[item["content"] for item in records],
            metadatas=[item["metadata"] for item in records],
            embeddings=[item["embedding"] for item in records],
        )

    def query(self, namespace: str, query_text: str, top_k: int) -> list[VectorHit]:
        """执行向量检索并把距离转换为可读分数。"""
        collection = self._get_collection(namespace)
        query_embedding = self.embedding_service.embed_text(query_text)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[VectorHit] = []
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # 距离越小越相似，这里映射到 (0,1] 区间方便后续融合。
            score = 1.0 / (1.0 + float(dist))
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    content=doc,
                    metadata=meta or {},
                    score=score,
                )
            )
        return hits

