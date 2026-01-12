"""
State store implementations for agent checkpoints and run state.

Provides:
- StateStore: Abstract interface
- InMemoryStateStore: For testing and simple use cases
- RedisStateStore: For production with Redis
- SQLiteStateStore: For persistent local storage
"""

from agent_runtime_core.state.base import StateStore
from agent_runtime_core.state.memory import InMemoryStateStore

__all__ = [
    "StateStore",
    "InMemoryStateStore",
    "get_state_store",
]


def get_state_store(backend: str = None, **kwargs) -> StateStore:
    """
    Factory function to get a state store.
    
    Args:
        backend: "memory", "redis", or "sqlite"
        **kwargs: Backend-specific configuration
        
    Returns:
        StateStore instance
    """
    from agent_runtime_core.config import get_config
    
    config = get_config()
    backend = backend or config.state_store_backend
    
    if backend == "memory":
        return InMemoryStateStore()
    
    elif backend == "redis":
        from agent_runtime_core.state.redis import RedisStateStore
        url = kwargs.get("url") or config.redis_url
        if not url:
            raise ValueError("redis_url is required for redis state store backend")
        return RedisStateStore(url=url)
    
    elif backend == "sqlite":
        from agent_runtime_core.state.sqlite import SQLiteStateStore
        path = kwargs.get("path") or config.sqlite_path or "agent_runtime.db"
        return SQLiteStateStore(path=path)
    
    else:
        raise ValueError(f"Unknown state store backend: {backend}")
