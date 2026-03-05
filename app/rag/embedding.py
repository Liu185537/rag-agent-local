from __future__ import annotations

import hashlib
import logging
import re

import httpx
import numpy as np

from app.core.config import Settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.Client(timeout=60.0)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = self._embed_with_ollama(text)
            if vector is None:
                vector = self._hashed_embedding(text)
            vectors.append(vector)
        return vectors

    def _embed_with_ollama(self, text: str) -> list[float] | None:
        if self.settings.llm_provider.lower() != "ollama":
            return None
        payload = {"model": self.settings.ollama_embed_model, "prompt": text}
        try:
            resp = self._client.post(f"{self.settings.ollama_base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if isinstance(embedding, list) and embedding:
                return [float(v) for v in embedding]
            return None
        except Exception as exc:
            logger.warning("Embedding request failed, using fallback embeddings: %s", exc)
            return None

    def _hashed_embedding(self, text: str) -> list[float]:
        dim = self.settings.fallback_embed_dim
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            tokens = [text[:64] if text else "__empty__"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

