from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    env: str = Field(default="dev", alias="RAG_AGENT_ENV")
    host: str = Field(default="127.0.0.1", alias="RAG_AGENT_HOST")
    port: int = Field(default=8008, alias="RAG_AGENT_PORT")

    sqlite_path: Path = Field(
        default=PROJECT_ROOT / "data" / "rag_agent.db", alias="RAG_AGENT_SQLITE_PATH"
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma", alias="RAG_AGENT_CHROMA_PATH"
    )

    llm_provider: str = Field(default="ollama", alias="RAG_AGENT_LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="RAG_AGENT_OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="qwen2.5:7b-instruct", alias="RAG_AGENT_OLLAMA_CHAT_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="RAG_AGENT_OLLAMA_EMBED_MODEL")

    retrieval_top_k: int = Field(default=5, alias="RAG_AGENT_RETRIEVAL_TOP_K")
    vector_top_k: int = Field(default=8, alias="RAG_AGENT_VECTOR_TOP_K")
    bm25_top_k: int = Field(default=8, alias="RAG_AGENT_BM25_TOP_K")
    fallback_embed_dim: int = Field(default=384, alias="RAG_AGENT_FALLBACK_EMBED_DIM")

    chunk_max_chars: int = 600
    chunk_overlap_chars: int = 80
    history_window: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_local_dirs(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_dirs()
    return settings

