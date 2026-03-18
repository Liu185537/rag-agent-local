from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    """返回 UTC 时间戳（ISO 格式）。"""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite 数据访问层。

    这里不使用 ORM，目的是让 SQL 与业务流程保持直观映射，
    便于演示和快速排查。
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """提供带自动提交和关闭的连接上下文。

        这样调用方只关注 SQL，不需要重复写 commit/close 模板代码。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        """初始化所有业务表。"""
        with self.connect() as conn:
            cursor = conn.cursor()
            # 会话表：记录会话归属的 namespace。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            # 消息表：保存用户/助手对话，以及引用与调试 trace。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    trace_json TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            # 用户画像：以 key-value 形式保存会话偏好或身份信息。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    session_id TEXT NOT NULL,
                    profile_key TEXT NOT NULL,
                    profile_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, profile_key)
                );
                """
            )
            # 文本分块元数据：供 BM25、本地排查与统计使用。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            # Agent 配置：按 namespace 保存助手名、描述与指令。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_config (
                    namespace TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def ensure_session(self, session_id: str, namespace: str) -> None:
        """确保会话存在，不存在则创建。"""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions(session_id, namespace, created_at)
                VALUES (?, ?, ?)
                """,
                (session_id, namespace, utc_now()),
            )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        trace: dict[str, Any] | None = None,
    ) -> None:
        """保存单条消息。

        citations 与 trace 以 JSON 字符串存储，保证结构灵活、表结构稳定。
        """
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, role, content, citations_json, trace_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(citations or [], ensure_ascii=False),
                    json.dumps(trace or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )

    def get_history(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """读取会话历史，并按时间正序返回。

        SQL 里按 `id DESC LIMIT N` 取“最近 N 条”更高效，
        取出后再 reverse 为前端更自然的时间顺序。
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, citations_json, trace_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        output: list[dict[str, Any]] = []
        # 查询按 id 倒序拿最新 N 条，再 reverse 回到自然对话顺序。
        for row in reversed(rows):
            output.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "citations": json.loads(row["citations_json"] or "[]"),
                    "trace": json.loads(row["trace_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return output

    def upsert_profile(self, session_id: str, key: str, value: str) -> None:
        """写入或更新用户画像字段。"""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profile(session_id, profile_key, profile_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, profile_key)
                DO UPDATE SET profile_value=excluded.profile_value, updated_at=excluded.updated_at
                """,
                (session_id, key, value, utc_now()),
            )

    def get_profile(self, session_id: str) -> dict[str, str]:
        """读取指定会话的全部画像字段。"""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT profile_key, profile_value
                FROM user_profile
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        return {row["profile_key"]: row["profile_value"] for row in rows}

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """批量写入/更新分块内容。

        使用 `ON CONFLICT(chunk_id)` 保证幂等：
        同一 chunk 再次入库时会覆盖旧内容与元数据。
        """
        if not chunks:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, namespace, doc_id, chunk_index, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id)
                DO UPDATE SET
                  namespace=excluded.namespace,
                  doc_id=excluded.doc_id,
                  chunk_index=excluded.chunk_index,
                  content=excluded.content,
                  metadata_json=excluded.metadata_json
                """,
                [
                    (
                        item["chunk_id"],
                        item["namespace"],
                        item["doc_id"],
                        item["chunk_index"],
                        item["content"],
                        json.dumps(item["metadata"], ensure_ascii=False),
                        utc_now(),
                    )
                    for item in chunks
                ],
            )

    def list_chunks(self, namespace: str) -> list[dict[str, Any]]:
        """列出某个知识空间下全部分块。"""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, chunk_index, content, metadata_json
                FROM chunks
                WHERE namespace = ?
                ORDER BY doc_id, chunk_index
                """,
                (namespace,),
            ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def recent_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """查询最近活跃会话，用于 Dashboard 展示。"""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  session_id,
                  MAX(created_at) AS last_message_at,
                  SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_turns,
                  SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS assistant_turns
                FROM messages
                GROUP BY session_id
                ORDER BY last_message_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "last_message_at": row["last_message_at"],
                "user_turns": row["user_turns"],
                "assistant_turns": row["assistant_turns"],
            }
            for row in rows
        ]

    def recent_assistant_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        """查询最近助手回复，用于 Dashboard 预览。"""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, content, citations_json, created_at
                FROM messages
                WHERE role = 'assistant'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "content": row["content"],
                "citations": json.loads(row["citations_json"] or "[]"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_agent_config(self, namespace: str) -> dict[str, str] | None:
        """读取某个 namespace 的 Agent 配置。"""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT namespace, name, description, instructions, updated_at
                FROM agent_config
                WHERE namespace = ?
                LIMIT 1
                """,
                (namespace,),
            ).fetchone()
        if row is None:
            return None
        return {
            "namespace": row["namespace"],
            "name": row["name"],
            "description": row["description"],
            "instructions": row["instructions"],
            "updated_at": row["updated_at"],
        }

    def upsert_agent_config(
        self, namespace: str, name: str, description: str, instructions: str
    ) -> dict[str, str]:
        """写入或更新 Agent 配置，并返回最新结果。"""
        updated_at = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_config(namespace, name, description, instructions, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace)
                DO UPDATE SET
                  name=excluded.name,
                  description=excluded.description,
                  instructions=excluded.instructions,
                  updated_at=excluded.updated_at
                """,
                (namespace, name, description, instructions, updated_at),
            )
        return {
            "namespace": namespace,
            "name": name,
            "description": description,
            "instructions": instructions,
            "updated_at": updated_at,
        }

    def delete_agent_config(self, namespace: str) -> bool:
        """删除配置，返回是否实际删除了记录。"""
        with self.connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM agent_config
                WHERE namespace = ?
                """,
                (namespace,),
            )
            return cursor.rowcount > 0

    def get_namespace_stats(self, namespace: str) -> dict[str, int]:
        """返回知识空间统计：分块数、会话数、文档数。

        该统计用于 demo 页面快速展示，不参与核心检索逻辑。
        """
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM chunks WHERE namespace = ?) AS chunk_count,
                  (SELECT COUNT(*) FROM sessions WHERE namespace = ?) AS session_count,
                  (SELECT COUNT(DISTINCT doc_id) FROM chunks WHERE namespace = ?) AS doc_count
                """,
                (namespace, namespace, namespace),
            ).fetchone()
        return {
            "chunk_count": int(row["chunk_count"] or 0),
            "session_count": int(row["session_count"] or 0),
            "doc_count": int(row["doc_count"] or 0),
        }
