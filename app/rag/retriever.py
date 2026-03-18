from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.core.database import Database
from app.rag.indexer import ChromaIndexer
from app.rag.reranker import LightweightReranker, tokenize


@dataclass
class RetrievedChunk:
    """统一的检索结果结构，供 API 与 Agent 共用。"""

    chunk_id: str
    content: str
    doc_id: str
    source: str
    vector_score: float
    bm25_score: float
    fused_score: float
    rerank_score: float
    metadata: dict[str, Any]


class HybridRetriever:
    """混合检索器：向量检索 + BM25 + RRF 融合 + 轻量重排。"""

    def __init__(self, settings: Settings, db: Database, indexer: ChromaIndexer):
        self.settings = settings
        self.db = db
        self.indexer = indexer
        self.reranker = LightweightReranker()

    def retrieve(self, query: str, namespace: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """执行完整混合检索并返回最终结果。

        输入：
        - query: 用户查询
        - namespace: 检索知识空间
        - top_k: 最终返回条数（可选）

        流程：
        1. 向量召回（语义相关）；
        2. BM25 召回（词法相关）；
        3. 用 RRF 融合两路结果；
        4. 对融合候选做轻量重排；
        5. 返回前 top_k 条。
        """
        final_top_k = top_k or self.settings.retrieval_top_k
        vector_hits = self.indexer.query(namespace, query, self.settings.vector_top_k)
        bm25_hits = self._bm25_search(query, namespace, self.settings.bm25_top_k)

        fused: dict[str, RetrievedChunk] = {}
        # RRF 常用平滑常量，避免 rank 前后差距过大。
        rrf_k = 60

        # 先合入向量结果。
        for rank, hit in enumerate(vector_hits, start=1):
            meta = hit.metadata or {}
            doc_id = str(meta.get("doc_id", "unknown_doc"))
            source = str(meta.get("source", "unknown_source"))
            fused_score = 1.0 / (rrf_k + rank)
            fused[hit.chunk_id] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                content=hit.content,
                doc_id=doc_id,
                source=source,
                vector_score=hit.score,
                bm25_score=0.0,
                fused_score=fused_score,
                rerank_score=0.0,
                metadata=meta,
            )

        # 再合入 BM25 结果，对重合 chunk 叠加融合分。
        for rank, (chunk, bm25_score) in enumerate(bm25_hits, start=1):
            chunk_id = chunk["chunk_id"]
            bonus = 1.0 / (rrf_k + rank)
            if chunk_id in fused:
                fused[chunk_id].bm25_score = bm25_score
                fused[chunk_id].fused_score += bonus
            else:
                meta = chunk.get("metadata", {})
                fused[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    content=chunk["content"],
                    doc_id=chunk["doc_id"],
                    source=str(meta.get("source", "unknown_source")),
                    vector_score=0.0,
                    bm25_score=bm25_score,
                    fused_score=bonus,
                    rerank_score=0.0,
                    metadata=meta,
                )

        # 先按融合分排序，缩小重排候选池，降低开销。
        pre_ranked = sorted(fused.values(), key=lambda x: x.fused_score, reverse=True)
        if not pre_ranked:
            return []

        # 候选池适当放大，给重排留出选择空间。
        rerank_pool = pre_ranked[: max(final_top_k * 4, 12)]
        bm25_max = max((item.bm25_score for item in rerank_pool), default=0.0)
        for item in rerank_pool:
            # 计算多路信号并合成为最终重排分。
            signal = self.reranker.score(
                query=query,
                chunk_text=item.content,
                semantic=item.vector_score,
                bm25=item.bm25_score,
                fused=item.fused_score,
            )
            item.rerank_score = self.reranker.combine(signal, bm25_max=bm25_max)

        reranked = sorted(rerank_pool, key=lambda x: x.rerank_score, reverse=True)
        return reranked[:final_top_k]

    def _bm25_search(
        self, query: str, namespace: str, top_k: int
    ) -> list[tuple[dict[str, Any], float]]:
        """执行 BM25 检索。

        注意：这里直接使用 SQLite 中持久化的 chunk 文本构建 BM25 语料，
        因此即使向量服务异常，词法检索仍可独立工作。
        """
        chunks = self.db.list_chunks(namespace)
        if not chunks:
            return []
        corpus_tokens = [tokenize(item["content"]) for item in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        out: list[tuple[dict[str, Any], float]] = []
        for idx, score in indexed[:top_k]:
            out.append((chunks[idx], float(score)))
        return out
