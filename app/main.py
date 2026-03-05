from __future__ import annotations

import html
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.agent.orchestrator import RagAgent
from app.agent.tools import ToolRegistry
from app.api.schemas import (
    AgentConfigRequest,
    AgentConfigResponse,
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    IngestDocument,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.core.config import get_settings
from app.core.database import Database
from app.core.logging import configure_logging
from app.core.observability import MetricsCollector
from app.llm.client import OllamaChatClient
from app.rag.chunker import chunk_text
from app.rag.document_parser import parse_document
from app.rag.embedding import EmbeddingService
from app.rag.indexer import ChromaIndexer
from app.rag.retriever import HybridRetriever


settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)

db = Database(settings.sqlite_path)
embedder = EmbeddingService(settings)
indexer = ChromaIndexer(settings, embedder)
retriever = HybridRetriever(settings=settings, db=db, indexer=indexer)
tools = ToolRegistry(db=db, retriever=retriever)
llm = OllamaChatClient(settings=settings)
agent = RagAgent(settings=settings, db=db, llm=llm, tools=tools)
metrics = MetricsCollector()

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="RAG-Agent Local", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    request_id = uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics.record(request.url.path, 500, latency_ms)
        logger.exception(
            "rid=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            500,
            latency_ms,
        )
        raise

    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics.record(request.url.path, status_code, latency_ms)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "rid=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        status_code,
        latency_ms,
    )
    return response


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.env}


def _default_agent_config(namespace: str) -> dict[str, str]:
    return {
        "namespace": namespace,
        "name": "知识库助手",
        "description": "",
        "instructions": "",
    }


@app.get("/api/v1/agent-config", response_model=AgentConfigResponse)
def get_agent_config(namespace: str = "default") -> AgentConfigResponse:
    ns = namespace.strip() or "default"
    config = db.get_agent_config(ns)
    if config is None:
        return AgentConfigResponse(namespace=ns, config=_default_agent_config(ns), source="default")
    return AgentConfigResponse(namespace=ns, config=config, source="storage")


@app.put("/api/v1/agent-config", response_model=AgentConfigResponse)
def upsert_agent_config(req: AgentConfigRequest) -> AgentConfigResponse:
    ns = req.namespace.strip() or "default"
    config = db.upsert_agent_config(
        namespace=ns,
        name=req.name.strip() or "知识库助手",
        description=req.description.strip(),
        instructions=req.instructions.strip(),
    )
    return AgentConfigResponse(namespace=ns, config=config, source="storage")


@app.delete("/api/v1/agent-config")
def delete_agent_config(namespace: str = "default") -> dict[str, object]:
    ns = namespace.strip() or "default"
    deleted = db.delete_agent_config(ns)
    return {"namespace": ns, "deleted": deleted}


def _ingest_documents(namespace: str, docs: list[IngestDocument]) -> tuple[int, int]:
    if not docs:
        raise HTTPException(status_code=400, detail="docs cannot be empty")

    index_records = []
    db_records = []

    for doc in docs:
        parts = chunk_text(
            text=doc.text,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        embeddings = embedder.embed_texts(parts)
        for idx, (part, embedding) in enumerate(zip(parts, embeddings)):
            chunk_id = f"{namespace}:{doc.doc_id}:{idx}"
            metadata = {
                "namespace": namespace,
                "doc_id": doc.doc_id,
                "chunk_index": idx,
                "source": doc.source,
                **doc.metadata,
            }
            index_records.append(
                {
                    "chunk_id": chunk_id,
                    "content": part,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )
            db_records.append(
                {
                    "chunk_id": chunk_id,
                    "namespace": namespace,
                    "doc_id": doc.doc_id,
                    "chunk_index": idx,
                    "content": part,
                    "metadata": metadata,
                }
            )

    indexer.upsert(namespace, index_records)
    db.upsert_chunks(db_records)
    return len(docs), len(index_records)


@app.post("/api/v1/knowledge/ingest", response_model=IngestResponse)
def ingest_knowledge(req: IngestRequest) -> IngestResponse:
    ingested_docs, ingested_chunks = _ingest_documents(req.namespace, req.docs)

    return IngestResponse(
        namespace=req.namespace,
        ingested_docs=ingested_docs,
        ingested_chunks=ingested_chunks,
    )


@app.post("/api/v1/knowledge/upload")
async def upload_knowledge(
    namespace: str = Form(default="default"),
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    source: str | None = Form(default=None),
) -> dict[str, object]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    file_name = file.filename or "upload.txt"
    try:
        text = parse_document(file_name=file_name, content_bytes=raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_doc_id = doc_id or Path(file_name).stem or f"doc_{uuid4().hex[:8]}"
    docs = [
        IngestDocument(
            doc_id=normalized_doc_id,
            text=text,
            source=source or file_name,
            metadata={
                "file_name": file_name,
                "content_type": file.content_type or "application/octet-stream",
                "upload_api": True,
            },
        )
    ]
    ingested_docs, ingested_chunks = _ingest_documents(namespace, docs)
    return {
        "namespace": namespace,
        "doc_id": normalized_doc_id,
        "file_name": file_name,
        "ingested_docs": ingested_docs,
        "ingested_chunks": ingested_chunks,
    }


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    chunks = retriever.retrieve(req.query, req.namespace, req.top_k)
    payload = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source": c.source,
            "content": c.content,
            "vector_score": c.vector_score,
            "bm25_score": c.bm25_score,
            "fused_score": c.fused_score,
            "rerank_score": c.rerank_score,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    return RetrieveResponse(namespace=req.namespace, query=req.query, chunks=payload)


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = req.session_id or str(uuid4())
    db.ensure_session(session_id=session_id, namespace=req.namespace)
    db.save_message(session_id=session_id, role="user", content=req.message)

    result = agent.run(
        session_id=session_id,
        namespace=req.namespace,
        user_input=req.message,
        top_k=req.top_k,
    )
    db.save_message(
        session_id=session_id,
        role="assistant",
        content=result.answer,
        citations=result.citations,
        trace=result.trace,
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        citations=result.citations,
        trace=result.trace,
    )


@app.get("/api/v1/sessions/{session_id}/history", response_model=HistoryResponse)
def history(session_id: str) -> HistoryResponse:
    rows = db.get_history(session_id=session_id, limit=100)
    return HistoryResponse(session_id=session_id, messages=rows)


@app.get("/api/v1/sessions/{session_id}/profile")
def profile(session_id: str) -> dict[str, object]:
    return {"session_id": session_id, "profile": db.get_profile(session_id)}


@app.get("/api/v1/metrics")
def api_metrics() -> dict[str, object]:
    return metrics.snapshot()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    snapshot = metrics.snapshot()
    sessions = db.recent_sessions(limit=10)
    messages = db.recent_assistant_messages(limit=10)

    session_rows = "\n".join(
        f"<tr><td>{html.escape(item['session_id'])}</td><td>{item['user_turns']}</td>"
        f"<td>{item['assistant_turns']}</td><td>{html.escape(str(item['last_message_at']))}</td></tr>"
        for item in sessions
    )
    message_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['session_id'])}</td>"
        f"<td>{html.escape(item['content'][:120])}</td>"
        f"<td>{len(item['citations'])}</td>"
        f"<td>{html.escape(str(item['created_at']))}</td>"
        "</tr>"
        for item in messages
    )
    top_path_rows = "\n".join(
        f"<tr><td>{html.escape(item['path'])}</td><td>{item['count']}</td></tr>"
        for item in snapshot["top_paths"]
    )

    return f"""
    <html>
      <head>
        <title>RAG-Agent Dashboard</title>
        <style>
          body {{ font-family: 'Segoe UI', sans-serif; margin: 24px; background: #f6f8fb; color: #111; }}
          .card {{ background: white; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
          table {{ width: 100%; border-collapse: collapse; }}
          th, td {{ border-bottom: 1px solid #eee; text-align: left; padding: 8px; vertical-align: top; }}
          h1, h2 {{ margin: 0 0 12px 0; }}
          .kpi {{ display: inline-block; margin-right: 24px; font-weight: 600; }}
        </style>
      </head>
      <body>
        <div class="card">
          <h1>RAG-Agent Local Dashboard</h1>
          <div class="kpi">Total Requests: {snapshot["total_requests"]}</div>
          <div class="kpi">Error Rate: {snapshot["error_rate"]:.2%}</div>
          <div class="kpi">Avg Latency: {snapshot["avg_latency_ms"]} ms</div>
          <div class="kpi">Updated: {html.escape(str(snapshot["last_updated_at"]))}</div>
        </div>
        <div class="card">
          <h2>Top Paths</h2>
          <table>
            <thead><tr><th>Path</th><th>Count</th></tr></thead>
            <tbody>{top_path_rows}</tbody>
          </table>
        </div>
        <div class="card">
          <h2>Recent Sessions</h2>
          <table>
            <thead><tr><th>Session ID</th><th>User Turns</th><th>Assistant Turns</th><th>Last Message</th></tr></thead>
            <tbody>{session_rows}</tbody>
          </table>
        </div>
        <div class="card">
          <h2>Recent Assistant Messages</h2>
          <table>
            <thead><tr><th>Session ID</th><th>Content Preview</th><th>Citations</th><th>Created At</th></tr></thead>
            <tbody>{message_rows}</tbody>
          </table>
        </div>
      </body>
    </html>
    """


PLAYGROUND_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "playground.html"


@app.get("/playground", response_class=HTMLResponse)
def playground() -> HTMLResponse:
    return HTMLResponse(PLAYGROUND_TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": "RAG-Agent Local",
        "docs": "/docs",
        "health": "/api/v1/health",
        "agent_config": "/api/v1/agent-config?namespace=demo",
        "metrics": "/api/v1/metrics",
        "dashboard": "/dashboard",
        "playground": "/playground",
        "note": "Local-only configuration, no company IPs or keys.",
    }

