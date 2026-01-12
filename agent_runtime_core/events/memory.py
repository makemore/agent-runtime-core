"""
In-memory event bus implementation.

Good for:
- Unit testing
- Local development
- Simple single-process scripts
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from uuid import UUID

from agent_runtime_core.events.base import EventBus, Event


class InMemoryEventBus(EventBus):
    """
    In-memory event bus implementation.
    
    Stores events in memory. Data is lost when the process exits.
    """
    
    def __init__(self):
        # run_id -> list of events
        self._events: dict[UUID, list[Event]] = defaultdict(list)
        # run_id -> list of subscriber queues
        self._subscribers: dict[UUID, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def publish(
        self,
        run_id: UUID,
        event_type: str,
        payload: dict,
    ) -> None:
        """Publish an event."""
        async with self._lock:
            events = self._events[run_id]
            sequence = len(events)
            
            event = Event(
                run_id=run_id,
                event_type=event_type,
                payload=payload,
                timestamp=datetime.now(timezone.utc),
                sequence=sequence,
            )
            
            events.append(event)
            
            # Notify subscribers
            for queue in self._subscribers[run_id]:
                await queue.put(event)
    
    async def subscribe(
        self,
        run_id: UUID,
    ) -> AsyncIterator[Event]:
        """Subscribe to events for a run."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        
        async with self._lock:
            self._subscribers[run_id].append(queue)
        
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers[run_id]:
                    self._subscribers[run_id].remove(queue)
    
    async def get_events(
        self,
        run_id: UUID,
        since_sequence: int = 0,
    ) -> list[Event]:
        """Get historical events for a run."""
        events = self._events.get(run_id, [])
        return [e for e in events if e.sequence >= since_sequence]
    
    def clear(self) -> None:
        """Clear all events. Useful for testing."""
        self._events.clear()
        self._subscribers.clear()
