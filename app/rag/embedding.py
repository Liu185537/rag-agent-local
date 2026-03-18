from __future__ import annotations

import hashlib
import logging
import re

import httpx
import numpy as np

from app.core.config import Settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    """向量化服务。

    优先调用 provider 对应 embedding 接口；
    若不可用，则回退到哈希向量，保证系统仍可演示检索流程。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.Client(timeout=60.0)

    def embed_text(self, text: str) -> list[float]:
        """向量化单条文本。"""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """向量化多条文本，失败时逐条回退。"""
        vectors: list[list[float]] = []
        provider = self.settings.llm_provider.lower().strip()
        for text in texts:
            if provider == "siliconflow":
                vector = self._embed_with_siliconflow(text)
            else:
                vector = self._embed_with_ollama(text)
            if vector is None:
                vector = self._hashed_embedding(text)
            vectors.append(vector)
        return vectors

    def _embed_with_ollama(self, text: str) -> list[float] | None:
        """调用 Ollama 生成 embedding，失败返回 None。"""
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

    def _embed_with_siliconflow(self, text: str) -> list[float] | None:
        """调用 SiliconFlow（OpenAI 兼容）/embeddings。"""
        if self.settings.llm_provider.lower() != "siliconflow":
            return None
        if not self.settings.siliconflow_api_key.strip():
            return None

        payload = {
            "model": self.settings.siliconflow_embed_model,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.siliconflow_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._client.post(
                f"{self.settings.siliconflow_base_url.rstrip('/')}/embeddings",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data") or []
            if not rows:
                return None
            embedding = rows[0].get("embedding")
            if isinstance(embedding, list) and embedding:
                return [float(v) for v in embedding]
            return None
        except Exception as exc:
            logger.warning("Embedding request failed (siliconflow), using fallback: %s", exc)
            return None

    def _hashed_embedding(self, text: str) -> list[float]:
        """哈希回退向量。

        原理：把 token 哈希到固定维度并做正负累加，最后归一化。
        该方法不追求语义效果，只保证离线可运行。
        """
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

