from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import JOBS_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(JOBS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    Path(JOBS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INT NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                csv_path TEXT NOT NULL DEFAULT '',
                preview_path TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                callback_url TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                files_deleted_at TIMESTAMP
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "user_id" not in columns:
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'"
            )
        if "updated_at" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN updated_at TIMESTAMP")
            conn.execute("UPDATE jobs SET updated_at = created_at WHERE updated_at IS NULL")
        if "files_deleted_at" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN files_deleted_at TIMESTAMP")
        if "callback_url" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN callback_url TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")


def create_job(
    job_id: str,
    user_id: str = "anonymous",
    callback_url: Optional[str] = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, status, progress, message, user_id, callback_url)
            VALUES (?, 'queued', 0, 'Видео загружено, ожидание обработки...', ?, ?)
            """,
            (job_id, user_id or "anonymous", callback_url or ""),
        )


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    csv_path: Optional[str] = None,
    preview_path: Optional[str] = None,
) -> None:
    fields: List[str] = []
    values: List[Any] = []

    for column, value in (
        ("status", status),
        ("progress", progress),
        ("message", message),
        ("csv_path", csv_path),
        ("preview_path", preview_path),
    ):
        if value is not None:
            fields.append(f"{column} = ?")
            values.append(value)

    if not fields:
        return

    fields.append("updated_at = CURRENT_TIMESTAMP")
    values.append(job_id)

    with _connect() as conn:
        conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            values,
        )


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 200))
    filters: List[str] = []
    params: List[Any] = []

    if user_id:
        filters.append("user_id = ?")
        params.append(user_id)

    if status:
        filters.append("status = ?")
        params.append(status)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM jobs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    return [dict(row) for row in rows]


def delete_job(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def list_jobs_for_analytics(days: int = 30) -> List[Dict[str, Any]]:
    days = max(1, min(days, 365))

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at ASC
            """,
            (f"-{days} days",),
        ).fetchall()

    return [dict(row) for row in rows]


def list_jobs_for_file_cleanup(retention_days: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status IN ('done', 'error')
              AND created_at < datetime('now', ?)
              AND files_deleted_at IS NULL
            ORDER BY created_at ASC
            """,
            (f"-{retention_days} days",),
        ).fetchall()

    return [dict(row) for row in rows]


def mark_job_files_deleted(job_id: str, message: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET message = ?,
                csv_path = '',
                preview_path = '',
                files_deleted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (message, job_id),
        )
