"""
SQLite queue implementation.

Good for:
- Local development with persistence
- Single-process deployments
- Testing with real database
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from agent_runtime_core.queue.base import RunQueue, QueuedRun, RunStatus


class SQLiteQueue(RunQueue):
    """
    SQLite-backed queue implementation.
    
    Stores runs in a local SQLite database.
    Uses row-level locking for claim operations.
    """
    
    def __init__(self, path: str = "agent_runtime.db", lease_ttl_seconds: int = 30):
        self.path = path
        self.lease_ttl_seconds = lease_ttl_seconds
        self._initialized = False
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    agent_key TEXT NOT NULL,
                    input TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'queued',
                    lease_owner TEXT DEFAULT '',
                    lease_expires_at TEXT,
                    cancel_requested_at TEXT,
                    output TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    available_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_status 
                ON runs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_available 
                ON runs(status, available_at)
            """)
            conn.commit()
        
        self._initialized = True
    
    async def enqueue(
        self,
        run_id: UUID,
        agent_key: str,
        input: dict,
        metadata: Optional[dict] = None,
        max_attempts: int = 3,
    ) -> QueuedRun:
        """Add a new run to the queue."""
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, agent_key, input, metadata, max_attempts,
                    status, created_at, available_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    agent_key,
                    json.dumps(input),
                    json.dumps(metadata or {}),
                    max_attempts,
                    RunStatus.QUEUED.value,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            conn.commit()
        
        return QueuedRun(
            run_id=run_id,
            agent_key=agent_key,
            attempt=1,
            lease_expires_at=now,
            input=input,
            metadata=metadata or {},
            max_attempts=max_attempts,
            status=RunStatus.QUEUED,
        )
    
    async def claim(
        self,
        worker_id: str,
        agent_keys: Optional[list[str]] = None,
        batch_size: int = 1,
    ) -> list[QueuedRun]:
        """Claim runs from the queue."""
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        lease_expires = now + timedelta(seconds=self.lease_ttl_seconds)
        
        with self._get_connection() as conn:
            # Build query
            query = """
                SELECT run_id, agent_key, input, metadata, max_attempts, attempt
                FROM runs
                WHERE status = ?
                AND available_at <= ?
            """
            params = [RunStatus.QUEUED.value, now.isoformat()]
            
            if agent_keys:
                placeholders = ",".join("?" * len(agent_keys))
                query += f" AND agent_key IN ({placeholders})"
                params.extend(agent_keys)
            
            query += f" ORDER BY created_at ASC LIMIT {batch_size}"
            
            rows = conn.execute(query, params).fetchall()
            
            claimed = []
            for row in rows:
                run_id = row["run_id"]
                
                # Try to claim with optimistic locking
                result = conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, lease_owner = ?, lease_expires_at = ?,
                        started_at = COALESCE(started_at, ?)
                    WHERE run_id = ? AND status = ?
                    """,
                    (
                        RunStatus.RUNNING.value,
                        worker_id,
                        lease_expires.isoformat(),
                        now.isoformat(),
                        run_id,
                        RunStatus.QUEUED.value,
                    ),
                )
                
                if result.rowcount > 0:
                    claimed.append(QueuedRun(
                        run_id=UUID(row["run_id"]),
                        agent_key=row["agent_key"],
                        attempt=row["attempt"],
                        lease_expires_at=lease_expires,
                        input=json.loads(row["input"]),
                        metadata=json.loads(row["metadata"]),
                        max_attempts=row["max_attempts"],
                        status=RunStatus.RUNNING,
                    ))
            
            conn.commit()
            return claimed
    
    async def extend_lease(self, run_id: UUID, worker_id: str, seconds: int) -> bool:
        """Extend the lease on a run."""
        self._ensure_initialized()
        new_expires = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        
        with self._get_connection() as conn:
            result = conn.execute(
                """
                UPDATE runs
                SET lease_expires_at = ?
                WHERE run_id = ? AND lease_owner = ? AND status = ?
                """,
                (
                    new_expires.isoformat(),
                    str(run_id),
                    worker_id,
                    RunStatus.RUNNING.value,
                ),
            )
            conn.commit()
            return result.rowcount > 0
    
    async def release(
        self,
        run_id: UUID,
        worker_id: str,
        success: bool,
        output: Optional[dict] = None,
        error: Optional[dict] = None,
    ) -> None:
        """Release a run after completion."""
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, lease_owner = '',
                    lease_expires_at = NULL, output = ?, error = ?
                WHERE run_id = ? AND lease_owner = ?
                """,
                (
                    RunStatus.SUCCEEDED.value if success else RunStatus.FAILED.value,
                    now.isoformat(),
                    json.dumps(output) if output else None,
                    json.dumps(error) if error else None,
                    str(run_id),
                    worker_id,
                ),
            )
            conn.commit()
    
    async def requeue_for_retry(
        self,
        run_id: UUID,
        worker_id: str,
        error: dict,
        delay_seconds: int = 0,
    ) -> bool:
        """Requeue a run for retry."""
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        available_at = now + timedelta(seconds=delay_seconds)
        
        with self._get_connection() as conn:
            # Get current attempt and max
            row = conn.execute(
                "SELECT attempt, max_attempts FROM runs WHERE run_id = ? AND lease_owner = ?",
                (str(run_id), worker_id),
            ).fetchone()
            
            if not row:
                return False
            
            if row["attempt"] >= row["max_attempts"]:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, error = ?, finished_at = ?,
                        lease_owner = '', lease_expires_at = NULL
                    WHERE run_id = ?
                    """,
                    (
                        RunStatus.FAILED.value,
                        json.dumps(error),
                        now.isoformat(),
                        str(run_id),
                    ),
                )
                conn.commit()
                return False
            
            conn.execute(
                """
                UPDATE runs
                SET status = ?, attempt = attempt + 1, error = ?,
                    lease_owner = '', lease_expires_at = NULL, available_at = ?
                WHERE run_id = ?
                """,
                (
                    RunStatus.QUEUED.value,
                    json.dumps(error),
                    available_at.isoformat(),
                    str(run_id),
                ),
            )
            conn.commit()
            return True
    
    async def cancel(self, run_id: UUID) -> bool:
        """Mark a run for cancellation."""
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            result = conn.execute(
                """
                UPDATE runs
                SET cancel_requested_at = ?
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (
                    now.isoformat(),
                    str(run_id),
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                ),
            )
            conn.commit()
            return result.rowcount > 0
    
    async def is_cancelled(self, run_id: UUID) -> bool:
        """Check if a run has been cancelled."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT cancel_requested_at FROM runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            
            return bool(row and row["cancel_requested_at"])
    
    async def recover_expired_leases(self) -> int:
        """Recover runs with expired leases."""
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            # Get expired runs
            rows = conn.execute(
                """
                SELECT run_id, attempt, max_attempts, agent_key
                FROM runs
                WHERE status = ? AND lease_expires_at < ?
                """,
                (RunStatus.RUNNING.value, now.isoformat()),
            ).fetchall()
            
            count = 0
            for row in rows:
                if row["attempt"] >= row["max_attempts"]:
                    conn.execute(
                        """
                        UPDATE runs
                        SET status = ?, finished_at = ?, error = ?,
                            lease_owner = '', lease_expires_at = NULL
                        WHERE run_id = ?
                        """,
                        (
                            RunStatus.TIMED_OUT.value,
                            now.isoformat(),
                            json.dumps({
                                "type": "LeaseExpired",
                                "message": "Worker lease expired without completion",
                                "retriable": False,
                            }),
                            row["run_id"],
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE runs
                        SET status = ?, attempt = attempt + 1,
                            lease_owner = '', lease_expires_at = NULL
                        WHERE run_id = ?
                        """,
                        (RunStatus.QUEUED.value, row["run_id"]),
                    )
                count += 1
            
            conn.commit()
            return count
    
    async def get_run(self, run_id: UUID) -> Optional[QueuedRun]:
        """Get a run by ID."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, agent_key, input, metadata, max_attempts,
                       attempt, status, lease_expires_at
                FROM runs WHERE run_id = ?
                """,
                (str(run_id),),
            ).fetchone()
            
            if not row:
                return None
            
            lease_expires = row["lease_expires_at"]
            
            return QueuedRun(
                run_id=UUID(row["run_id"]),
                agent_key=row["agent_key"],
                attempt=row["attempt"],
                lease_expires_at=(
                    datetime.fromisoformat(lease_expires) if lease_expires
                    else datetime.now(timezone.utc)
                ),
                input=json.loads(row["input"]),
                metadata=json.loads(row["metadata"]),
                max_attempts=row["max_attempts"],
                status=RunStatus(row["status"]),
            )
