"""
Redis Streams-backed queue with consumer groups.

Good for:
- Production deployments
- Multi-process/distributed setups
- High throughput
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from agent_runtime_core.queue.base import RunQueue, QueuedRun, RunStatus


class RedisQueue(RunQueue):
    """
    Redis Streams-backed queue implementation.

    Uses consumer groups for distributed processing.
    Run state is stored in Redis hashes.
    """

    STREAM_KEY = "agent_runtime:queue"
    RUNS_KEY = "agent_runtime:runs"
    GROUP_NAME = "agent_workers"

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        lease_ttl_seconds: int = 30,
        stream_key: Optional[str] = None,
        runs_key: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        self.url = url
        self.lease_ttl_seconds = lease_ttl_seconds
        self.stream_key = stream_key or self.STREAM_KEY
        self.runs_key = runs_key or self.RUNS_KEY
        self.group_name = group_name or self.GROUP_NAME
        self._client = None

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisQueue. "
                    "Install with: pip install agent_runtime[redis]"
                )
            self._client = redis.from_url(self.url)
            # Ensure consumer group exists
            try:
                await self._client.xgroup_create(
                    self.stream_key, self.group_name, id="0", mkstream=True
                )
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    raise
        return self._client

    def _run_key(self, run_id: UUID) -> str:
        """Get Redis key for a run."""
        return f"{self.runs_key}:{run_id}"

    async def enqueue(
        self,
        run_id: UUID,
        agent_key: str,
        input: dict,
        metadata: Optional[dict] = None,
        max_attempts: int = 3,
    ) -> QueuedRun:
        """Add a new run to the queue."""
        client = await self._get_client()
        now = datetime.now(timezone.utc)
        
        run_data = {
            "run_id": str(run_id),
            "agent_key": agent_key,
            "input": json.dumps(input),
            "metadata": json.dumps(metadata or {}),
            "max_attempts": str(max_attempts),
            "attempt": "1",
            "status": RunStatus.QUEUED.value,
            "lease_owner": "",
            "lease_expires_at": "",
            "cancel_requested_at": "",
            "created_at": now.isoformat(),
            "started_at": "",
            "finished_at": "",
            "output": "",
            "error": "",
        }
        
        # Store run data
        await client.hset(self._run_key(run_id), mapping=run_data)
        
        # Add to stream
        await client.xadd(
            self.stream_key,
            {"run_id": str(run_id), "agent_key": agent_key},
        )
        
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
        """Claim runs from the stream using consumer groups."""
        client = await self._get_client()
        now = datetime.now(timezone.utc)
        lease_expires = now + timedelta(seconds=self.lease_ttl_seconds)

        # Read from consumer group
        messages = await client.xreadgroup(
            self.group_name,
            worker_id,
            {self.stream_key: ">"},
            count=batch_size,
            block=1000,  # 1 second block
        )

        if not messages:
            return []

        claimed = []
        for stream_name, stream_messages in messages:
            for msg_id, data in stream_messages:
                run_id_str = data.get(b"run_id", data.get("run_id"))
                if isinstance(run_id_str, bytes):
                    run_id_str = run_id_str.decode()
                run_id = UUID(run_id_str)
                
                agent_key = data.get(b"agent_key", data.get("agent_key"))
                if isinstance(agent_key, bytes):
                    agent_key = agent_key.decode()

                # Filter by agent_keys if specified
                if agent_keys and agent_key not in agent_keys:
                    await client.xack(self.stream_key, self.group_name, msg_id)
                    continue

                # Get and update run data
                run_key = self._run_key(run_id)
                run_data = await client.hgetall(run_key)
                
                if not run_data:
                    await client.xack(self.stream_key, self.group_name, msg_id)
                    continue
                
                # Decode bytes if needed
                run_data = {
                    (k.decode() if isinstance(k, bytes) else k): 
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in run_data.items()
                }
                
                status = run_data.get("status", "")
                if status not in [RunStatus.QUEUED.value, RunStatus.RUNNING.value]:
                    await client.xack(self.stream_key, self.group_name, msg_id)
                    continue
                
                # Check if already claimed
                if status == RunStatus.RUNNING.value:
                    existing_expires = run_data.get("lease_expires_at", "")
                    if existing_expires:
                        expires_dt = datetime.fromisoformat(existing_expires)
                        if expires_dt > now:
                            await client.xack(self.stream_key, self.group_name, msg_id)
                            continue
                
                # Claim the run
                updates = {
                    "status": RunStatus.RUNNING.value,
                    "lease_owner": worker_id,
                    "lease_expires_at": lease_expires.isoformat(),
                }
                if not run_data.get("started_at"):
                    updates["started_at"] = now.isoformat()
                
                await client.hset(run_key, mapping=updates)
                await client.xack(self.stream_key, self.group_name, msg_id)
                
                claimed.append(QueuedRun(
                    run_id=run_id,
                    agent_key=agent_key,
                    attempt=int(run_data.get("attempt", 1)),
                    lease_expires_at=lease_expires,
                    input=json.loads(run_data.get("input", "{}")),
                    metadata=json.loads(run_data.get("metadata", "{}")),
                    max_attempts=int(run_data.get("max_attempts", 3)),
                    status=RunStatus.RUNNING,
                ))

        return claimed

    async def extend_lease(self, run_id: UUID, worker_id: str, seconds: int) -> bool:
        """Extend lease in Redis."""
        client = await self._get_client()
        run_key = self._run_key(run_id)
        
        run_data = await client.hgetall(run_key)
        if not run_data:
            return False
        
        # Decode
        run_data = {
            (k.decode() if isinstance(k, bytes) else k): 
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in run_data.items()
        }
        
        if run_data.get("lease_owner") != worker_id:
            return False
        if run_data.get("status") != RunStatus.RUNNING.value:
            return False
        
        new_expires = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        await client.hset(run_key, "lease_expires_at", new_expires.isoformat())
        return True

    async def release(
        self,
        run_id: UUID,
        worker_id: str,
        success: bool,
        output: Optional[dict] = None,
        error: Optional[dict] = None,
    ) -> None:
        """Release run after completion."""
        client = await self._get_client()
        run_key = self._run_key(run_id)
        now = datetime.now(timezone.utc)
        
        updates = {
            "status": RunStatus.SUCCEEDED.value if success else RunStatus.FAILED.value,
            "finished_at": now.isoformat(),
            "lease_owner": "",
            "lease_expires_at": "",
        }
        if output:
            updates["output"] = json.dumps(output)
        if error:
            updates["error"] = json.dumps(error)
        
        await client.hset(run_key, mapping=updates)

    async def requeue_for_retry(
        self,
        run_id: UUID,
        worker_id: str,
        error: dict,
        delay_seconds: int = 0,
    ) -> bool:
        """Requeue for retry."""
        client = await self._get_client()
        run_key = self._run_key(run_id)
        
        run_data = await client.hgetall(run_key)
        if not run_data:
            return False
        
        run_data = {
            (k.decode() if isinstance(k, bytes) else k): 
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in run_data.items()
        }
        
        if run_data.get("lease_owner") != worker_id:
            return False
        
        attempt = int(run_data.get("attempt", 1))
        max_attempts = int(run_data.get("max_attempts", 3))
        
        if attempt >= max_attempts:
            await client.hset(run_key, mapping={
                "status": RunStatus.FAILED.value,
                "error": json.dumps(error),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "lease_owner": "",
                "lease_expires_at": "",
            })
            return False
        
        await client.hset(run_key, mapping={
            "status": RunStatus.QUEUED.value,
            "attempt": str(attempt + 1),
            "error": json.dumps(error),
            "lease_owner": "",
            "lease_expires_at": "",
        })
        
        # Re-add to stream (with delay if needed - simplified, no delay support)
        agent_key = run_data.get("agent_key", "")
        await client.xadd(
            self.stream_key,
            {"run_id": str(run_id), "agent_key": agent_key},
        )
        return True

    async def cancel(self, run_id: UUID) -> bool:
        """Mark run for cancellation."""
        client = await self._get_client()
        run_key = self._run_key(run_id)
        
        run_data = await client.hgetall(run_key)
        if not run_data:
            return False
        
        run_data = {
            (k.decode() if isinstance(k, bytes) else k): 
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in run_data.items()
        }
        
        status = run_data.get("status", "")
        if status not in [RunStatus.QUEUED.value, RunStatus.RUNNING.value]:
            return False
        
        await client.hset(
            run_key, 
            "cancel_requested_at", 
            datetime.now(timezone.utc).isoformat()
        )
        return True

    async def is_cancelled(self, run_id: UUID) -> bool:
        """Check if cancellation was requested."""
        client = await self._get_client()
        run_key = self._run_key(run_id)
        
        cancel_at = await client.hget(run_key, "cancel_requested_at")
        if isinstance(cancel_at, bytes):
            cancel_at = cancel_at.decode()
        return bool(cancel_at)

    async def recover_expired_leases(self) -> int:
        """Recover runs with expired leases."""
        client = await self._get_client()
        now = datetime.now(timezone.utc)
        
        # Scan for all run keys
        count = 0
        async for key in client.scan_iter(f"{self.runs_key}:*"):
            run_data = await client.hgetall(key)
            if not run_data:
                continue
            
            run_data = {
                (k.decode() if isinstance(k, bytes) else k): 
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in run_data.items()
            }
            
            if run_data.get("status") != RunStatus.RUNNING.value:
                continue
            
            lease_expires = run_data.get("lease_expires_at", "")
            if not lease_expires:
                continue
            
            expires_dt = datetime.fromisoformat(lease_expires)
            if expires_dt > now:
                continue
            
            # Lease expired
            attempt = int(run_data.get("attempt", 1))
            max_attempts = int(run_data.get("max_attempts", 3))
            run_id = run_data.get("run_id", "")
            agent_key = run_data.get("agent_key", "")
            
            if attempt >= max_attempts:
                await client.hset(key, mapping={
                    "status": RunStatus.TIMED_OUT.value,
                    "finished_at": now.isoformat(),
                    "error": json.dumps({
                        "type": "LeaseExpired",
                        "message": "Worker lease expired without completion",
                        "retriable": False,
                    }),
                    "lease_owner": "",
                    "lease_expires_at": "",
                })
            else:
                await client.hset(key, mapping={
                    "status": RunStatus.QUEUED.value,
                    "attempt": str(attempt + 1),
                    "lease_owner": "",
                    "lease_expires_at": "",
                })
                # Re-add to stream
                await client.xadd(
                    self.stream_key,
                    {"run_id": run_id, "agent_key": agent_key},
                )
            
            count += 1
        
        return count

    async def get_run(self, run_id: UUID) -> Optional[QueuedRun]:
        """Get a run by ID."""
        client = await self._get_client()
        run_key = self._run_key(run_id)
        
        run_data = await client.hgetall(run_key)
        if not run_data:
            return None
        
        run_data = {
            (k.decode() if isinstance(k, bytes) else k): 
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in run_data.items()
        }
        
        lease_expires = run_data.get("lease_expires_at", "")
        
        return QueuedRun(
            run_id=UUID(run_data["run_id"]),
            agent_key=run_data.get("agent_key", ""),
            attempt=int(run_data.get("attempt", 1)),
            lease_expires_at=(
                datetime.fromisoformat(lease_expires) if lease_expires 
                else datetime.now(timezone.utc)
            ),
            input=json.loads(run_data.get("input", "{}")),
            metadata=json.loads(run_data.get("metadata", "{}")),
            max_attempts=int(run_data.get("max_attempts", 3)),
            status=RunStatus(run_data.get("status", "queued")),
        )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
