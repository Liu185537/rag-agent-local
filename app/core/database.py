from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
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
        with self.connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM agent_config
                WHERE namespace = ?
                """,
                (namespace,),
            )
            return cursor.rowcount > 0
