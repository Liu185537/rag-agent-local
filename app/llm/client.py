from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.config import Settings


logger = logging.getLogger(__name__)
UNAVAILABLE_PREFIX = "[local-llm-unavailable]"


class OllamaChatClient:
    """聊天客户端（支持 Ollama / SiliconFlow）。

    约定：
    - 正常模式返回模型文本；
    - 失败时返回可识别前缀，便于上层执行降级逻辑。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.Client(timeout=90.0)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """调用聊天接口；json_mode 用于 planner 的结构化输出。"""
        provider = self.settings.llm_provider.lower().strip()
        if provider == "ollama":
            return self._chat_with_ollama(messages, temperature=temperature, json_mode=json_mode)
        if provider == "siliconflow":
            return self._chat_with_siliconflow(messages, temperature=temperature, json_mode=json_mode)

        if json_mode:
            return self._default_json_plan()
        return f"{UNAVAILABLE_PREFIX} unsupported provider: {self.settings.llm_provider}"

    def _chat_with_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> str:
        """调用 Ollama /api/chat。"""
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
            logger.warning("LLM request failed (ollama): %s", exc)
            if json_mode:
                return self._default_json_plan()
            return f"{UNAVAILABLE_PREFIX} {exc}"

    def _chat_with_siliconflow(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> str:
        """调用 SiliconFlow（OpenAI 兼容）/chat/completions。"""
        if not self.settings.siliconflow_api_key.strip():
            if json_mode:
                return self._default_json_plan()
            return f"{UNAVAILABLE_PREFIX} missing siliconflow api key"

        payload: dict[str, Any] = {
            "model": self.settings.siliconflow_chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.settings.siliconflow_api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._client.post(
                f"{self.settings.siliconflow_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            return str(choices[0].get("message", {}).get("content", "")).strip()
        except Exception as exc:
            logger.warning("LLM request failed (siliconflow): %s", exc)
            if json_mode:
                return self._default_json_plan()
            return f"{UNAVAILABLE_PREFIX} {exc}"

    @staticmethod
    def _default_json_plan() -> str:
        """模型不可用时的 planner 兜底 JSON。"""
        return json.dumps(
            {
                "intent": "knowledge_qa",
                "needs_retrieval": True,
                "profile_updates": [],
                "reason": "fallback planner without model",
            }
        )

