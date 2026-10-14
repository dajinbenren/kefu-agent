"""
db.py —— SQLite 会话持久化

职责一句话：会话和消息落盘，刷新页面、重启服务都不丢。

选型：SQLite = Python 自带标准库，单文件零部署。
小型公司一个 app.db 文件就够，养不起也用不上 MySQL。
两张表：sessions（一个会话）→ messages（会话里的每条消息）。
"""

import json
import sqlite3

DB_PATH = "app.db"


def _conn() -> sqlite3.Connection:
    # 每次操作开新连接：FastAPI 同步接口跑在线程池里，
    # "每个请求一个连接"是最省心的姿势（SQLite 本身极快，无开销焦虑）
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row  # 查询结果能按列名取值
    return c


def init_db() -> None:
    """建表（IF NOT EXISTS = 启动多少次都安全）。lifespan 里调用。"""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,              -- user / assistant
                content TEXT,
                meta TEXT DEFAULT '',   -- 来源/工具轨迹等附件，JSON 字符串
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
        """)


def save_message(session_id: str, role: str, content: str, meta: dict | None = None) -> None:
    """存一条消息（顺便确保会话存在）。"""
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,))
        c.execute(
            "INSERT INTO messages(session_id, role, content, meta) VALUES (?,?,?,?)",
            (session_id, role, content, json.dumps(meta or {}, ensure_ascii=False)),
        )


def load_messages(session_id: str, limit: int = 12) -> list[dict]:
    """取最近 limit 条消息（按时间正序返回）。给模型当上下文用。"""
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, meta FROM messages "
            "WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def list_sessions() -> list[dict]:
    """会话列表：每个会话带首条用户问题当标题（侧边栏用）。"""
    with _conn() as c:
        rows = c.execute("""
            SELECT s.id, s.created_at,
                (SELECT content FROM messages m
                 WHERE m.session_id = s.id AND m.role = 'user'
                 ORDER BY m.id LIMIT 1) AS first_question
            FROM sessions s ORDER BY s.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]
