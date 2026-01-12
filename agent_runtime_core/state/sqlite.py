"""
SQLite state store implementation.

Good for:
- Local development with persistence
- Single-process deployments
- Testing with real database
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from agent_runtime_core.state.base import StateStore, Checkpoint


class SQLiteStateStore(StateStore):
    """
    SQLite-backed state store.
    
    Stores checkpoints in a local SQLite database.
    Supports both file-based and in-memory databases.
    """
    
    def __init__(self, path: str = "agent_runtime.db"):
        """
        Initialize SQLite state store.
        
        Args:
            path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.path = path
        self._initialized = False
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _ensure_initialized(self):
        """Ensure the database schema exists."""
        if self._initialized:
            return
        
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, seq)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id 
                ON checkpoints(run_id)
            """)
            conn.commit()
        
        self._initialized = True
    
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (run_id, seq, state, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(checkpoint.run_id),
                    checkpoint.seq,
                    json.dumps(checkpoint.state),
                    checkpoint.created_at.isoformat(),
                ),
            )
            conn.commit()
    
    async def get_latest_checkpoint(self, run_id: UUID) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a run."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, seq, state, created_at
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY seq DESC
                LIMIT 1
                """,
                (str(run_id),),
            ).fetchone()
            
            if not row:
                return None
            
            return Checkpoint(
                run_id=UUID(row["run_id"]),
                seq=row["seq"],
                state=json.loads(row["state"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
    
    async def get_checkpoints(self, run_id: UUID) -> list[Checkpoint]:
        """Get all checkpoints for a run."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT run_id, seq, state, created_at
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (str(run_id),),
            ).fetchall()
            
            return [
                Checkpoint(
                    run_id=UUID(row["run_id"]),
                    seq=row["seq"],
                    state=json.loads(row["state"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]
    
    async def get_next_seq(self, run_id: UUID) -> int:
        """Get the next sequence number for a run."""
        latest = await self.get_latest_checkpoint(run_id)
        return (latest.seq + 1) if latest else 0
    
    async def delete_checkpoints(self, run_id: UUID) -> int:
        """Delete all checkpoints for a run."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE run_id = ?",
                (str(run_id),),
            )
            conn.commit()
            return cursor.rowcount
