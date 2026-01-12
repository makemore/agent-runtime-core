"""
Queue implementations for agent run scheduling.

Provides:
- RunQueue: Abstract interface
- QueuedRun: Data structure for queued runs
- InMemoryQueue: For testing and simple use cases
- RedisQueue: For production with Redis
- SQLiteQueue: For persistent local storage
"""

from agent_runtime_core.queue.base import RunQueue, QueuedRun
from agent_runtime_core.queue.memory import InMemoryQueue

__all__ = [
    "RunQueue",
    "QueuedRun",
    "InMemoryQueue",
    "get_queue",
]


def get_queue(backend: str = None, **kwargs) -> RunQueue:
    """
    Factory function to get a run queue.
    
    Args:
        backend: "memory", "redis", or "sqlite"
        **kwargs: Backend-specific configuration
        
    Returns:
        RunQueue instance
    """
    from agent_runtime_core.config import get_config
    
    config = get_config()
    backend = backend or config.queue_backend
    
    if backend == "memory":
        return InMemoryQueue()
    
    elif backend == "redis":
        from agent_runtime_core.queue.redis import RedisQueue
        url = kwargs.get("url") or config.redis_url
        if not url:
            raise ValueError("redis_url is required for redis queue backend")
        return RedisQueue(url=url)
    
    elif backend == "sqlite":
        from agent_runtime_core.queue.sqlite import SQLiteQueue
        path = kwargs.get("path") or config.sqlite_path or "agent_runtime.db"
        return SQLiteQueue(path=path)
    
    else:
        raise ValueError(f"Unknown queue backend: {backend}")
