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
    """Agent 执行结果：回答、引用及调试轨迹。"""

    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]


class RagAgent:
    """RAG Agent 编排器。

    主流程：
    1. 读取历史与画像；
    2. planner 决定是否检索/更新画像；
    3. 调用检索工具；
    4. 结合上下文生成答案并产出 trace。
    """

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
        """执行单轮 Agent。

        输入：
        - session_id: 会话 ID，用于读取历史和画像。
        - namespace: 知识空间，用于约束检索范围。
        - user_input: 用户本轮问题。
        - top_k: 可选覆盖默认召回条数。

        处理：
        1. 读取历史与画像；
        2. planner 产出计划；
        3. 必要时更新画像；
        4. 必要时检索知识；
        5. 生成回答并整理引用。

        输出：
        - AgentResult(answer, citations, trace)
        """
        history = self.db.get_history(session_id=session_id, limit=self.settings.history_window)
        profile = self.tools.get_profile(session_id)
        plan = self._plan(user_input=user_input, history=history, profile=profile)

        # 若 planner 给出画像更新，则先写回再继续后续流程。
        updates = plan.get("profile_updates", [])
        if updates:
            self.tools.update_profile(session_id, updates)
            profile = self.tools.get_profile(session_id)

        contexts: list[RetrievedChunk] = []
        # planner 偶尔会把普通问答误判为无需检索，这里做一层稳健兜底。
        needs_retrieval = bool(plan.get("needs_retrieval", True))
        intent = str(plan.get("intent", "")).lower().strip()
        if (not needs_retrieval) and intent != "profile_update":
            needs_retrieval = True

        # planner 判断需要检索时，调用知识检索工具。
        if needs_retrieval:
            contexts = self.tools.retrieve_knowledge(
                query=user_input,
                namespace=namespace,
                top_k=top_k or self.settings.retrieval_top_k,
            )

        # 对知识问答类请求，如果首次检索为空，再做一次兜底检索。
        if (not contexts) and intent == "knowledge_qa":
            contexts = self.tools.retrieve_knowledge(
                query=user_input,
                namespace=namespace,
                top_k=top_k or self.settings.retrieval_top_k,
            )

        # 生成最终回答。
        answer = self._generate_answer(
            user_input=user_input,
            profile=profile,
            history=history,
            contexts=contexts,
        )

        citations = self._build_citations(contexts)

        # trace 便于调试：记录 planner 输出、工具调用和检索规模。
        trace = {
            "planner_output": plan,
            "tools_used": [
                "get_profile",
                "update_profile" if updates else "no_profile_update",
                "retrieve_knowledge" if needs_retrieval else "no_retrieval",
            ],
            "retrieved_chunks": len(contexts),
            "citation_docs": len(citations),
        }
        return AgentResult(answer=answer, citations=citations, trace=trace)

    def _build_citations(self, contexts: list[RetrievedChunk]) -> list[dict[str, Any]]:
        """构建引用列表，并按文档去重。

        说明：
        - 检索结果是 chunk 级别，同一文档可能命中多个 chunk；
        - 前端更关注“来自哪个文档”，这里按 doc_id 仅保留每个文档的首条结果。
        """
        seen_docs: set[str] = set()
        citations: list[dict[str, Any]] = []
        for item in contexts:
            doc_key = item.doc_id.strip() if item.doc_id else item.chunk_id
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)
            citations.append(
                {
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "source": item.source,
                    "score": round(item.fused_score, 6),
                }
            )
        return citations

    def _plan(
        self,
        user_input: str,
        history: list[dict[str, Any]],
        profile: dict[str, str],
    ) -> dict[str, Any]:
        """生成 planner 结构化结果。

        返回约定字段：
        - intent: 意图类型
        - needs_retrieval: 是否需要检索
        - profile_updates: 画像更新列表
        - reason: 规划原因说明

        若模型输出不合法 JSON，会自动走 `_heuristic_plan`，
        保证系统不会因为 planner 失败而中断。
        """
        # 低成本快速路径：明显是知识问答或画像更新时，跳过一次 planner LLM 调用。
        fast_plan = self._fast_path_plan(user_input)
        if fast_plan is not None:
            return fast_plan

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
            # 模型返回非 JSON 时，退回启发式规则，保证可用性。
            return self._heuristic_plan(user_input)

    def _fast_path_plan(self, user_input: str) -> dict[str, Any] | None:
        """快速规划：在高置信场景下直接返回计划，减少一次 LLM 往返延迟。"""
        lower = user_input.lower().strip()
        if not lower:
            return None

        name_match = re.search(r"(?:my name is|call me)\s+([a-zA-Z][a-zA-Z0-9_-]{1,31})", lower)
        if name_match:
            return {
                "intent": "profile_update",
                "needs_retrieval": False,
                "profile_updates": [{"key": "name", "value": name_match.group(1)}],
                "reason": "fast-path profile extraction",
            }

        # 问句/咨询语气默认按知识问答处理，稳定性更好且可避免额外 planner 调用。
        knowledge_markers = ["?", "？", "how", "what", "why", "when", "where", "如何", "怎么", "什么"]
        if any(token in lower for token in knowledge_markers):
            return {
                "intent": "knowledge_qa",
                "needs_retrieval": True,
                "profile_updates": [],
                "reason": "fast-path knowledge qa",
            }

        return None

    def _heuristic_plan(self, user_input: str) -> dict[str, Any]:
        """启发式兜底规划。

        当前规则很小：
        - 若识别到 "my name is/call me"，更新画像并跳过检索；
        - 否则默认走知识问答并开启检索。
        """
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
        """根据上下文生成最终回答。

        关键点：
        - 检索结果会带编号，便于模型输出 [1]/[2] 这种引用标记；
        - 当本地模型不可用时，返回可读的降级答案，避免接口报错。
        """
        if not contexts:
            context_text = "No retrieved context."
        else:
            context_lines = []
            # 控制送入生成模型的上下文大小，降低延迟和 token 开销。
            max_context_chunks = 3
            max_context_chars = 420
            for idx, item in enumerate(contexts[:max_context_chunks], start=1):
                # 预先编号，便于模型按 [1][2] 方式引用。
                context_lines.append(
                    f"[{idx}] source={item.source}, doc_id={item.doc_id}, chunk_id={item.chunk_id}\n"
                    f"{item.content[:max_context_chars]}"
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
            # LLM 不可用时，给出可理解的检索兜底回答。
            if contexts:
                first = contexts[0]
                return (
                    "Local model is unavailable, fallback answer from top context:\n"
                    f"{first.content[:280]}...\n"
                    "Please start Ollama to enable full agent generation."
                )
            return "Local model is unavailable and no context was retrieved."
        return raw

