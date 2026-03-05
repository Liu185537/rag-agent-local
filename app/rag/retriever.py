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
    def __init__(self, settings: Settings, db: Database, indexer: ChromaIndexer):
        self.settings = settings
        self.db = db
        self.indexer = indexer
        self.reranker = LightweightReranker()

    def retrieve(self, query: str, namespace: str, top_k: int | None = None) -> list[RetrievedChunk]:
        final_top_k = top_k or self.settings.retrieval_top_k
        vector_hits = self.indexer.query(namespace, query, self.settings.vector_top_k)
        bm25_hits = self._bm25_search(query, namespace, self.settings.bm25_top_k)

        fused: dict[str, RetrievedChunk] = {}
        rrf_k = 60

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

        pre_ranked = sorted(fused.values(), key=lambda x: x.fused_score, reverse=True)
        if not pre_ranked:
            return []

        rerank_pool = pre_ranked[: max(final_top_k * 4, 12)]
        bm25_max = max((item.bm25_score for item in rerank_pool), default=0.0)
        for item in rerank_pool:
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
