from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.config import Settings


logger = logging.getLogger(__name__)
UNAVAILABLE_PREFIX = "[local-llm-unavailable]"


class OllamaChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.Client(timeout=90.0)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        if self.settings.llm_provider.lower() != "ollama":
            if json_mode:
                return self._default_json_plan()
            return f"{UNAVAILABLE_PREFIX} unsupported provider: {self.settings.llm_provider}"

        payload: dict[str, Any] = {
            "model": self.settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = self._client.post(f"{self.settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.warning("LLM request failed: %s", exc)
            if json_mode:
                return self._default_json_plan()
            return f"{UNAVAILABLE_PREFIX} {exc}"

    @staticmethod
    def _default_json_plan() -> str:
        return json.dumps(
            {
                "intent": "knowledge_qa",
                "needs_retrieval": True,
                "profile_updates": [],
                "reason": "fallback planner without model",
            }
        )

