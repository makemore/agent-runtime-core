"""
Redis state store implementation.

Good for:
- Production deployments
- Multi-process/distributed setups
- Automatic TTL-based cleanup
"""

import json
from datetime import datetime
from typing import Optional
from uuid import UUID

from agent_runtime_core.state.base import StateStore, Checkpoint


class RedisStateStore(StateStore):
    """
    Redis-backed state store.
    
    Stores checkpoints in Redis with optional TTL.
    Uses sorted sets for efficient retrieval by sequence number.
    """
    
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "agent_runtime:state:",
        ttl_seconds: int = 3600 * 24,  # 24 hours default
    ):
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._client = None
    
    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisStateStore. "
                    "Install with: pip install agent_runtime[redis]"
                )
            self._client = redis.from_url(self.url)
        return self._client
    
    def _key(self, run_id: UUID) -> str:
        """Get Redis key for a run's checkpoints."""
        return f"{self.prefix}{run_id}"
    
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        client = await self._get_client()
        key = self._key(checkpoint.run_id)
        
        # Serialize checkpoint
        data = json.dumps(checkpoint.to_dict())
        
        # Add to sorted set with seq as score
        await client.zadd(key, {data: checkpoint.seq})
        
        # Set TTL on the key
        await client.expire(key, self.ttl_seconds)
    
    async def get_latest_checkpoint(self, run_id: UUID) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a run."""
        client = await self._get_client()
        key = self._key(run_id)
        
        # Get highest scored item (latest seq)
        results = await client.zrevrange(key, 0, 0)
        if not results:
            return None
        
        data = json.loads(results[0])
        return Checkpoint.from_dict(data)
    
    async def get_checkpoints(self, run_id: UUID) -> list[Checkpoint]:
        """Get all checkpoints for a run."""
        client = await self._get_client()
        key = self._key(run_id)
        
        # Get all items ordered by seq
        results = await client.zrange(key, 0, -1)
        
        return [Checkpoint.from_dict(json.loads(r)) for r in results]
    
    async def get_next_seq(self, run_id: UUID) -> int:
        """Get the next sequence number for a run."""
        latest = await self.get_latest_checkpoint(run_id)
        return (latest.seq + 1) if latest else 0
    
    async def delete_checkpoints(self, run_id: UUID) -> int:
        """Delete all checkpoints for a run."""
        client = await self._get_client()
        key = self._key(run_id)
        
        count = await client.zcard(key)
        await client.delete(key)
        return count
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
