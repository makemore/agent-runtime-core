"""
Redis-backed event bus using pub/sub and streams.

Good for:
- Production deployments
- Multi-process/distributed setups
- Real-time streaming
"""

import asyncio
import json
from typing import AsyncIterator, Optional
from uuid import UUID

from agent_runtime_core.events.base import EventBus, Event


class RedisEventBus(EventBus):
    """
    Redis-backed event bus implementation.

    Uses Redis Streams for event storage and pub/sub for real-time notifications.
    """

    STREAM_PREFIX = "agent_runtime:events:"
    CHANNEL_PREFIX = "agent_runtime:notify:"

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        event_ttl_seconds: int = 3600 * 6,  # 6 hours
    ):
        self.url = url
        self.event_ttl_seconds = event_ttl_seconds
        self._client = None

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisEventBus. "
                    "Install with: pip install agent_runtime[redis]"
                )
            self._client = redis.from_url(self.url)
        return self._client

    def _stream_key(self, run_id: UUID) -> str:
        """Get Redis stream key for a run."""
        return f"{self.STREAM_PREFIX}{run_id}"

    def _channel_key(self, run_id: UUID) -> str:
        """Get Redis pub/sub channel for a run."""
        return f"{self.CHANNEL_PREFIX}{run_id}"

    async def publish(self, event: Event) -> None:
        """Publish event to Redis."""
        client = await self._get_client()

        # Add to stream
        stream_key = self._stream_key(event.run_id)
        await client.xadd(
            stream_key,
            {"data": json.dumps(event.to_dict())},
        )

        # Set TTL on stream
        await client.expire(stream_key, self.event_ttl_seconds)

        # Notify subscribers
        channel_key = self._channel_key(event.run_id)
        await client.publish(channel_key, str(event.seq))

    async def subscribe(
        self,
        run_id: UUID,
        from_seq: int = 0,
        check_complete: Optional[callable] = None,
    ) -> AsyncIterator[Event]:
        """
        Subscribe to events using pub/sub for notifications.
        
        Args:
            run_id: Run to subscribe to
            from_seq: Start from this sequence number
            check_complete: Optional async callable that returns True when run is complete
        """
        client = await self._get_client()
        pubsub = client.pubsub()
        channel_key = self._channel_key(run_id)

        await pubsub.subscribe(channel_key)

        try:
            # First, get any existing events
            events = await self.get_events(run_id, from_seq=from_seq)
            current_seq = from_seq

            for event in events:
                yield event
                current_seq = event.seq + 1

            # Then listen for new events
            while True:
                # Check if run is complete
                if check_complete and await check_complete():
                    # Get any final events
                    final_events = await self.get_events(run_id, from_seq=current_seq)
                    for event in final_events:
                        yield event
                    break

                # Wait for notification with timeout
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0,
                    )
                    if message:
                        # Get new events
                        new_events = await self.get_events(run_id, from_seq=current_seq)
                        for event in new_events:
                            yield event
                            current_seq = event.seq + 1
                except asyncio.TimeoutError:
                    continue

        finally:
            await pubsub.unsubscribe(channel_key)
            await pubsub.close()

    async def get_events(
        self,
        run_id: UUID,
        from_seq: int = 0,
        to_seq: Optional[int] = None,
    ) -> list[Event]:
        """Get events from Redis stream."""
        client = await self._get_client()
        stream_key = self._stream_key(run_id)

        # Read from stream
        messages = await client.xrange(stream_key)

        events = []
        for msg_id, data in messages:
            data_bytes = data.get(b"data", data.get("data"))
            if isinstance(data_bytes, bytes):
                data_bytes = data_bytes.decode()
            event_data = json.loads(data_bytes)
            event = Event.from_dict(event_data)

            if event.seq < from_seq:
                continue
            if to_seq is not None and event.seq > to_seq:
                continue

            events.append(event)

        return sorted(events, key=lambda e: e.seq)

    async def get_next_seq(self, run_id: UUID) -> int:
        """Get next sequence number from Redis."""
        client = await self._get_client()
        stream_key = self._stream_key(run_id)

        # Check Redis stream
        messages = await client.xrevrange(stream_key, count=1)
        if messages:
            msg_id, data = messages[0]
            data_bytes = data.get(b"data", data.get("data"))
            if isinstance(data_bytes, bytes):
                data_bytes = data_bytes.decode()
            event_data = json.loads(data_bytes)
            return event_data["seq"] + 1

        return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
