from __future__ import annotations

import re
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    """轻量分词：优先英文/数字 token，兜底按非空字符切分。"""
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
    """本地轻量重排器。

    组合四类信号：
    1. 向量语义分；
    2. BM25 词法分；
    3. RRF 融合分；
    4. 查询词覆盖率。
    """

    def score(self, query: str, chunk_text: str, semantic: float, bm25: float, fused: float) -> RerankSignal:
        """计算单个候选分块的基础重排信号。"""
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
        """将多路信号加权为最终重排分。

        这里会先做 BM25 归一化，避免不同语料规模导致分值量级偏差。
        """
        bm25_norm = signal.bm25_score / bm25_max if bm25_max > 0 else 0.0
        return (
            0.40 * signal.semantic_score
            + 0.20 * bm25_norm
            + 0.25 * signal.coverage_score
            + 0.15 * signal.fused_score
        )

