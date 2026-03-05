from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.agent.tools import ToolRegistry
from app.core.config import Settings
from app.core.database import Database
from app.llm.client import UNAVAILABLE_PREFIX, OllamaChatClient
from app.rag.retriever import RetrievedChunk


@dataclass
class AgentResult:
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]


class RagAgent:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        llm: OllamaChatClient,
        tools: ToolRegistry,
    ):
        self.settings = settings
        self.db = db
        self.llm = llm
        self.tools = tools

    def run(
        self,
        session_id: str,
        namespace: str,
        user_input: str,
        top_k: int | None = None,
    ) -> AgentResult:
        history = self.db.get_history(session_id=session_id, limit=self.settings.history_window)
        profile = self.tools.get_profile(session_id)
        plan = self._plan(user_input=user_input, history=history, profile=profile)

        updates = plan.get("profile_updates", [])
        if updates:
            self.tools.update_profile(session_id, updates)
            profile = self.tools.get_profile(session_id)

        contexts: list[RetrievedChunk] = []
        if bool(plan.get("needs_retrieval", True)):
            contexts = self.tools.retrieve_knowledge(
                query=user_input,
                namespace=namespace,
                top_k=top_k or self.settings.retrieval_top_k,
            )

        answer = self._generate_answer(
            user_input=user_input,
            profile=profile,
            history=history,
            contexts=contexts,
        )

        citations = [
            {
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "source": item.source,
                "score": round(item.fused_score, 6),
            }
            for item in contexts
        ]

        trace = {
            "planner_output": plan,
            "tools_used": [
                "get_profile",
                "update_profile" if updates else "no_profile_update",
                "retrieve_knowledge" if contexts else "no_retrieval",
            ],
            "retrieved_chunks": len(contexts),
        }
        return AgentResult(answer=answer, citations=citations, trace=trace)

    def _plan(
        self,
        user_input: str,
        history: list[dict[str, Any]],
        profile: dict[str, str],
    ) -> dict[str, Any]:
        planner_prompt = (
            "You are the planner for a RAG agent.\n"
            "Return strict JSON with keys: intent, needs_retrieval, profile_updates, reason.\n"
            "intent must be one of: knowledge_qa, profile_update, chitchat.\n"
            "profile_updates must be a list of objects with key/value.\n"
            "If the user asks a domain question, set needs_retrieval=true."
        )
        messages = [
            {"role": "system", "content": planner_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_input": user_input,
                        "history": history[-2:],
                        "profile": profile,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self.llm.chat(messages, temperature=0.0, json_mode=True)
        try:
            plan = json.loads(raw)
            if "intent" not in plan:
                raise ValueError("planner missing intent")
            plan.setdefault("needs_retrieval", True)
            plan.setdefault("profile_updates", [])
            plan.setdefault("reason", "planner")
            return plan
        except Exception:
            return self._heuristic_plan(user_input)

    def _heuristic_plan(self, user_input: str) -> dict[str, Any]:
        lower = user_input.lower()
        updates: list[dict[str, str]] = []
        name_match = re.search(r"(?:my name is|call me)\s+([a-zA-Z][a-zA-Z0-9_-]{1,31})", lower)
        if name_match:
            updates.append({"key": "name", "value": name_match.group(1)})

        if updates:
            return {
                "intent": "profile_update",
                "needs_retrieval": False,
                "profile_updates": updates,
                "reason": "regex profile extraction fallback",
            }
        return {
            "intent": "knowledge_qa",
            "needs_retrieval": True,
            "profile_updates": [],
            "reason": "default fallback",
        }

    def _generate_answer(
        self,
        user_input: str,
        profile: dict[str, str],
        history: list[dict[str, Any]],
        contexts: list[RetrievedChunk],
    ) -> str:
        if not contexts:
            context_text = "No retrieved context."
        else:
            context_lines = []
            for idx, item in enumerate(contexts, start=1):
                context_lines.append(
                    f"[{idx}] source={item.source}, doc_id={item.doc_id}, chunk_id={item.chunk_id}\n{item.content}"
                )
            context_text = "\n\n".join(context_lines)

        system_prompt = (
            "You are a pragmatic RAG assistant.\n"
            "Rules:\n"
            "1) If context is provided, answer based on context.\n"
            "2) If context is insufficient, say what is missing.\n"
            "3) Include citation markers like [1], [2] when using context.\n"
            "4) Keep answer concise and precise."
        )
        user_payload = {
            "question": user_input,
            "profile": profile,
            "recent_history": history[-4:],
            "context": context_text,
        }
        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            json_mode=False,
        )

        if raw.startswith(UNAVAILABLE_PREFIX):
            if contexts:
                first = contexts[0]
                return (
                    "Local model is unavailable, fallback answer from top context:\n"
                    f"{first.content[:280]}...\n"
                    "Please start Ollama to enable full agent generation."
                )
            return "Local model is unavailable and no context was retrieved."
        return raw

