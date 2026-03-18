from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    """单个待入库文档。"""

    doc_id: str
    text: str
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """知识入库请求体。"""

    namespace: str = "default"
    docs: list[IngestDocument]


class IngestResponse(BaseModel):
    """知识入库返回体。"""

    namespace: str
    ingested_docs: int
    ingested_chunks: int


class ChatRequest(BaseModel):
    """聊天请求体。"""

    message: str
    namespace: str = "default"
    session_id: str | None = None
    top_k: int | None = None


class ChatResponse(BaseModel):
    """聊天响应：答案、引用和调试轨迹。"""

    session_id: str
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]


class RetrieveRequest(BaseModel):
    """纯检索请求体（不走生成）。"""

    query: str
    namespace: str = "default"
    top_k: int | None = None


class RetrieveResponse(BaseModel):
    """纯检索返回体。"""

    namespace: str
    query: str
    chunks: list[dict[str, Any]]


class HistoryResponse(BaseModel):
    """会话历史返回体。"""

    session_id: str
    messages: list[dict[str, Any]]


class AgentConfigRequest(BaseModel):
    """Agent 配置写入请求体。"""

    namespace: str = "default"
    name: str = "Knowledge Assistant"
    description: str = ""
    instructions: str = ""


class AgentConfigResponse(BaseModel):
    """Agent 配置读取/写入返回体。"""

    namespace: str
    config: dict[str, str]
    source: str
