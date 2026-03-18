from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录，用于拼接默认本地数据路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """集中管理服务配置。

    设计目标：
    - 提供开箱即用的本地默认值；
    - 允许通过 .env 或系统环境变量覆盖；
    - 避免配置分散在多个文件中。
    """

    # 服务基本信息。
    env: str = Field(default="dev", alias="RAG_AGENT_ENV")
    host: str = Field(default="127.0.0.1", alias="RAG_AGENT_HOST")
    port: int = Field(default=8008, alias="RAG_AGENT_PORT")

    # 本地持久化目录：SQLite 与 Chroma。
    sqlite_path: Path = Field(
        default=PROJECT_ROOT / "data" / "rag_agent.db", alias="RAG_AGENT_SQLITE_PATH"
    )
    chroma_path: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma", alias="RAG_AGENT_CHROMA_PATH"
    )

    # 模型相关配置：支持 ollama / siliconflow。
    llm_provider: str = Field(default="ollama", alias="RAG_AGENT_LLM_PROVIDER")

    # Ollama 配置。
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="RAG_AGENT_OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="qwen2.5:7b-instruct", alias="RAG_AGENT_OLLAMA_CHAT_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="RAG_AGENT_OLLAMA_EMBED_MODEL")

    # SiliconFlow（OpenAI 兼容接口）配置。
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1", alias="RAG_AGENT_SILICONFLOW_BASE_URL"
    )
    siliconflow_api_key: str = Field(default="", alias="RAG_AGENT_SILICONFLOW_API_KEY")
    siliconflow_chat_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", alias="RAG_AGENT_SILICONFLOW_CHAT_MODEL"
    )
    siliconflow_embed_model: str = Field(
        default="BAAI/bge-m3", alias="RAG_AGENT_SILICONFLOW_EMBED_MODEL"
    )

    # 检索参数：向量召回、BM25 召回与最终返回条数。
    retrieval_top_k: int = Field(default=5, alias="RAG_AGENT_RETRIEVAL_TOP_K")
    vector_top_k: int = Field(default=8, alias="RAG_AGENT_VECTOR_TOP_K")
    bm25_top_k: int = Field(default=8, alias="RAG_AGENT_BM25_TOP_K")
    fallback_embed_dim: int = Field(default=384, alias="RAG_AGENT_FALLBACK_EMBED_DIM")

    # 文本切分与对话历史窗口。
    chunk_max_chars: int = 600
    chunk_overlap_chars: int = 80
    history_window: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_local_dirs(self) -> None:
        """启动前确保本地目录存在，避免首次运行报错。"""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一配置实例。

    使用缓存可以避免多次解析 .env，同时保证各模块读取到同一份配置。
    """
    settings = Settings()
    settings.ensure_local_dirs()
    return settings

