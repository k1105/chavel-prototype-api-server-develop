"""
会話履歴管理

SQLite で session_id 単位で会話履歴を保存・取得
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# データベースパス
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "dialog.sqlite"


def init_db():
    """データベースとテーブルを初期化"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # sessions テーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # messages テーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            pos INTEGER,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, timestamp)
    """)

    conn.commit()
    conn.close()


def generate_session_id() -> str:
    """新しいセッション ID を生成"""
    return str(uuid.uuid4())


def create_session(session_id: Optional[str] = None) -> str:
    """セッションを作成"""
    if session_id is None:
        session_id = generate_session_id()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()

    # すでに存在する場合はスキップ
    cursor.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,)
    )
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now)
        )
        conn.commit()

    conn.close()
    return session_id


def add_message(
    session_id: str,
    role: str,
    content: str,
    pos: Optional[int] = None
):
    """メッセージを追加"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()

    cursor.execute(
        """INSERT INTO messages (session_id, role, content, pos, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, role, content, pos, now)
    )

    # セッションの updated_at を更新
    cursor.execute(
        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id)
    )

    conn.commit()
    conn.close()


def get_recent_messages(
    session_id: str,
    limit: int = 12
) -> List[Dict[str, Any]]:
    """最新 N 件のメッセージを取得"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """SELECT role, content, pos, timestamp
           FROM messages
           WHERE session_id = ?
           ORDER BY timestamp DESC
           LIMIT ?""",
        (session_id, limit)
    )

    messages = []
    for row in cursor.fetchall():
        messages.append({
            "role": row[0],
            "content": row[1],
            "pos": row[2],
            "timestamp": row[3]
        })

    conn.close()

    # 時系列順に並び替え（DESC で取得したので逆順に）
    return list(reversed(messages))


def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """セッション全体の履歴を取得"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """SELECT role, content, pos, timestamp
           FROM messages
           WHERE session_id = ?
           ORDER BY timestamp ASC""",
        (session_id,)
    )

    messages = []
    for row in cursor.fetchall():
        messages.append({
            "role": row[0],
            "content": row[1],
            "pos": row[2],
            "timestamp": row[3]
        })

    conn.close()
    return messages


def session_exists(session_id: str) -> bool:
    """セッションが存在するか確認"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?",
        (session_id,)
    )

    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# 起動時に初期化
init_db()
