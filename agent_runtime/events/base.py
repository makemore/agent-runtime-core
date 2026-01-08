"""
Abstract base class for event bus implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional
from uuid import UUID


@dataclass
class Event:
    """An event emitted by an agent run."""
    
    run_id: UUID
    event_type: str
    payload: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0


class EventBus(ABC):
    """
    Abstract interface for event bus implementations.
    
    Event buses handle:
    - Publishing events from agent runs
    - Subscribing to events for a run
    - Event persistence (optional)
    """
    
    @abstractmethod
    async def publish(
        self,
        run_id: UUID,
        event_type: str,
        payload: dict,
    ) -> None:
        """
        Publish an event.
        
        Args:
            run_id: Run that emitted the event
            event_type: Type of event
            payload: Event data
        """
        ...
    
    @abstractmethod
    async def subscribe(
        self,
        run_id: UUID,
    ) -> AsyncIterator[Event]:
        """
        Subscribe to events for a run.
        
        Args:
            run_id: Run to subscribe to
            
        Yields:
            Events as they are published
        """
        ...
    
    @abstractmethod
    async def get_events(
        self,
        run_id: UUID,
        since_sequence: int = 0,
    ) -> list[Event]:
        """
        Get historical events for a run.
        
        Args:
            run_id: Run to get events for
            since_sequence: Only return events after this sequence
            
        Returns:
            List of events
        """
        ...
    
    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass
