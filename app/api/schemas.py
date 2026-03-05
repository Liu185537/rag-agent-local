from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    doc_id: str
    text: str
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    namespace: str = "default"
    docs: list[IngestDocument]


class IngestResponse(BaseModel):
    namespace: str
    ingested_docs: int
    ingested_chunks: int


class ChatRequest(BaseModel):
    message: str
    namespace: str = "default"
    session_id: str | None = None
    top_k: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]


class RetrieveRequest(BaseModel):
    query: str
    namespace: str = "default"
    top_k: int | None = None


class RetrieveResponse(BaseModel):
    namespace: str
    query: str
    chunks: list[dict[str, Any]]


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, Any]]


class AgentConfigRequest(BaseModel):
    namespace: str = "default"
    name: str = "Knowledge Assistant"
    description: str = ""
    instructions: str = ""


class AgentConfigResponse(BaseModel):
    namespace: str
    config: dict[str, str]
    source: str
