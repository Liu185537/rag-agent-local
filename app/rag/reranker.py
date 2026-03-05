from __future__ import annotations

import re
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if tokens:
        return tokens
    return [ch for ch in text if ch.strip()]


@dataclass
class RerankSignal:
    semantic_score: float
    bm25_score: float
    fused_score: float
    coverage_score: float


class LightweightReranker:
    """
    A local reranker that combines:
    - semantic score from vector retrieval
    - lexical BM25 score
    - RRF fused score
    - token coverage between query and chunk
    """

    def score(self, query: str, chunk_text: str, semantic: float, bm25: float, fused: float) -> RerankSignal:
        q = set(tokenize(query))
        c = set(tokenize(chunk_text))
        coverage = len(q & c) / max(1, len(q))
        return RerankSignal(
            semantic_score=max(0.0, semantic),
            bm25_score=max(0.0, bm25),
            fused_score=max(0.0, fused),
            coverage_score=max(0.0, coverage),
        )

    def combine(self, signal: RerankSignal, bm25_max: float) -> float:
        bm25_norm = signal.bm25_score / bm25_max if bm25_max > 0 else 0.0
        return (
            0.40 * signal.semantic_score
            + 0.20 * bm25_norm
            + 0.25 * signal.coverage_score
            + 0.15 * signal.fused_score
        )

