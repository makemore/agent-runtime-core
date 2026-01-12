"""
In-memory queue implementation.

Good for:
- Unit testing
- Local development
- Simple single-process scripts
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

from agent_runtime_core.queue.base import RunQueue, QueuedRun


@dataclass
class QueueEntry:
    """Internal queue entry with lease info."""
    run: QueuedRun
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    cancelled: bool = False
    completed: bool = False
    output: Optional[dict] = None
    error: Optional[dict] = None


class InMemoryQueue(RunQueue):
    """
    In-memory queue implementation.
    
    Stores runs in memory. Data is lost when the process exits.
    """
    
    def __init__(self, max_retries: int = 3):
        self._entries: dict[UUID, QueueEntry] = {}
        self._queue: deque[UUID] = deque()
        self._lock = asyncio.Lock()
        self._max_retries = max_retries
    
    async def enqueue(
        self,
        run_id: UUID,
        agent_key: str,
        input: dict,
        metadata: Optional[dict] = None,
        priority: int = 0,
    ) -> None:
        """Add a run to the queue."""
        async with self._lock:
            run = QueuedRun(
                run_id=run_id,
                agent_key=agent_key,
                input=input,
                metadata=metadata or {},
                priority=priority,
            )
            self._entries[run_id] = QueueEntry(run=run)
            self._queue.append(run_id)
    
    async def claim(
        self,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> Optional[QueuedRun]:
        """Claim the next available run."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            # Find an available run
            for _ in range(len(self._queue)):
                run_id = self._queue.popleft()
                entry = self._entries.get(run_id)
                
                if entry is None or entry.completed or entry.cancelled:
                    continue
                
                # Check if lease expired
                if entry.lease_owner and entry.lease_expires_at:
                    if entry.lease_expires_at > now:
                        # Still leased, put back
                        self._queue.append(run_id)
                        continue
                
                # Claim it
                entry.lease_owner = worker_id
                entry.lease_expires_at = now + timedelta(seconds=lease_seconds)
                return entry.run
            
            return None
    
    async def release(
        self,
        run_id: UUID,
        worker_id: str,
        success: bool,
        output: Optional[dict] = None,
        error: Optional[dict] = None,
    ) -> None:
        """Release a claimed run."""
        async with self._lock:
            entry = self._entries.get(run_id)
            if entry is None:
                return
            
            if entry.lease_owner != worker_id:
                return
            
            entry.completed = True
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.output = output
            entry.error = error
    
    async def extend_lease(
        self,
        run_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        """Extend the lease on a run."""
        async with self._lock:
            entry = self._entries.get(run_id)
            if entry is None:
                return False
            
            if entry.lease_owner != worker_id:
                return False
            
            entry.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            return True
    
    async def is_cancelled(self, run_id: UUID) -> bool:
        """Check if a run has been cancelled."""
        entry = self._entries.get(run_id)
        return entry.cancelled if entry else False
    
    async def cancel(self, run_id: UUID) -> bool:
        """Cancel a run."""
        async with self._lock:
            entry = self._entries.get(run_id)
            if entry is None or entry.completed:
                return False
            
            entry.cancelled = True
            return True
    
    async def requeue_for_retry(
        self,
        run_id: UUID,
        worker_id: str,
        error: dict,
        delay_seconds: int = 0,
    ) -> bool:
        """Requeue a failed run for retry."""
        async with self._lock:
            entry = self._entries.get(run_id)
            if entry is None:
                return False
            
            if entry.lease_owner != worker_id:
                return False
            
            if entry.run.attempt >= self._max_retries:
                return False
            
            # Increment attempt and requeue
            entry.run.attempt += 1
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.error = error
            
            # Add back to queue
            self._queue.append(run_id)
            return True
    
    def clear(self) -> None:
        """Clear all entries. Useful for testing."""
        self._entries.clear()
        self._queue.clear()
