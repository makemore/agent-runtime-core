"""
SQLite event bus implementation.

Good for:
- Local development with persistence
- Single-process deployments
- Testing with real database
"""

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from uuid import UUID

from agent_runtime_core.events.base import EventBus, Event


class SQLiteEventBus(EventBus):
    """
    SQLite-backed event bus implementation.
    
    Stores events in a local SQLite database.
    Uses polling for subscriptions (no real-time push).
    """
    
    def __init__(self, path: str = "agent_runtime.db"):
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
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(run_id, seq)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_run_id 
                ON events(run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_run_seq 
                ON events(run_id, seq)
            """)
            conn.commit()
        
        self._initialized = True
    
    async def publish(self, event: Event) -> None:
        """Publish an event."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events (run_id, seq, event_type, payload, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(event.run_id),
                    event.seq,
                    event.event_type,
                    json.dumps(event.payload),
                    event.timestamp.isoformat(),
                ),
            )
            conn.commit()
    
    async def subscribe(
        self,
        run_id: UUID,
        from_seq: int = 0,
        check_complete: Optional[callable] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[Event]:
        """
        Subscribe to events for a run.
        
        Uses polling since SQLite doesn't support real-time notifications.
        """
        current_seq = from_seq
        
        while True:
            # Get new events
            events = await self.get_events(run_id, from_seq=current_seq)
            
            for event in events:
                yield event
                current_seq = event.seq + 1
            
            # Check if run is complete
            if check_complete and await check_complete():
                break
            
            # Poll interval
            await asyncio.sleep(poll_interval)
    
    async def get_events(
        self,
        run_id: UUID,
        from_seq: int = 0,
        to_seq: Optional[int] = None,
    ) -> list[Event]:
        """Get historical events for a run."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            query = """
                SELECT run_id, seq, event_type, payload, timestamp
                FROM events
                WHERE run_id = ? AND seq >= ?
            """
            params = [str(run_id), from_seq]
            
            if to_seq is not None:
                query += " AND seq <= ?"
                params.append(to_seq)
            
            query += " ORDER BY seq ASC"
            
            rows = conn.execute(query, params).fetchall()
            
            return [
                Event(
                    run_id=UUID(row["run_id"]),
                    seq=row["seq"],
                    event_type=row["event_type"],
                    payload=json.loads(row["payload"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                for row in rows
            ]
    
    async def get_next_seq(self, run_id: UUID) -> int:
        """Get the next sequence number for a run."""
        self._ensure_initialized()
        
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(seq) as max_seq FROM events WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            
            max_seq = row["max_seq"]
            return (max_seq + 1) if max_seq is not None else 0
