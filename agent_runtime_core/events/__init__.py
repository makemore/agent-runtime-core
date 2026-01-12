"""
Event bus implementations for agent communication.

Provides:
- EventBus: Abstract interface
- Event: Event data structure
- InMemoryEventBus: For testing and simple use cases
- RedisEventBus: For production with pub/sub
- SQLiteEventBus: For persistent local storage
"""

from agent_runtime_core.events.base import EventBus, Event
from agent_runtime_core.events.memory import InMemoryEventBus

__all__ = [
    "EventBus",
    "Event",
    "InMemoryEventBus",
    "get_event_bus",
]


def get_event_bus(backend: str = None, **kwargs) -> EventBus:
    """
    Factory function to get an event bus.
    
    Args:
        backend: "memory", "redis", or "sqlite"
        **kwargs: Backend-specific configuration
        
    Returns:
        EventBus instance
    """
    from agent_runtime_core.config import get_config
    
    config = get_config()
    backend = backend or config.event_bus_backend
    
    if backend == "memory":
        return InMemoryEventBus()
    
    elif backend == "redis":
        from agent_runtime_core.events.redis import RedisEventBus
        url = kwargs.get("url") or config.redis_url
        if not url:
            raise ValueError("redis_url is required for redis event bus backend")
        return RedisEventBus(url=url, **kwargs)
    
    elif backend == "sqlite":
        from agent_runtime_core.events.sqlite import SQLiteEventBus
        path = kwargs.get("path") or config.sqlite_path or "agent_runtime.db"
        return SQLiteEventBus(path=path)
    
    else:
        raise ValueError(f"Unknown event bus backend: {backend}")
