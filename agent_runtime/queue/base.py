"""
Abstract base class for run queue implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID


@dataclass
class QueuedRun:
    """A run waiting in the queue."""
    
    run_id: UUID
    agent_key: str
    input: dict
    metadata: dict = field(default_factory=dict)
    priority: int = 0
    attempt: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None


class RunQueue(ABC):
    """
    Abstract interface for run queue implementations.
    
    Queues handle:
    - Enqueueing new runs
    - Claiming runs for processing
    - Lease management
    - Retries
    """
    
    @abstractmethod
    async def enqueue(
        self,
        run_id: UUID,
        agent_key: str,
        input: dict,
        metadata: Optional[dict] = None,
        priority: int = 0,
    ) -> None:
        """
        Add a run to the queue.
        
        Args:
            run_id: Unique run identifier
            agent_key: Agent to handle the run
            input: Input data for the run
            metadata: Optional metadata
            priority: Priority (higher = more urgent)
        """
        ...
    
    @abstractmethod
    async def claim(
        self,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> Optional[QueuedRun]:
        """
        Claim the next available run.
        
        Args:
            worker_id: ID of the claiming worker
            lease_seconds: How long to hold the lease
            
        Returns:
            QueuedRun if one is available, None otherwise
        """
        ...
    
    @abstractmethod
    async def release(
        self,
        run_id: UUID,
        worker_id: str,
        success: bool,
        output: Optional[dict] = None,
        error: Optional[dict] = None,
    ) -> None:
        """
        Release a claimed run.
        
        Args:
            run_id: Run to release
            worker_id: Worker releasing the run
            success: Whether the run succeeded
            output: Output data (if success)
            error: Error info (if failure)
        """
        ...
    
    @abstractmethod
    async def extend_lease(
        self,
        run_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        """
        Extend the lease on a run.
        
        Args:
            run_id: Run to extend
            worker_id: Worker holding the lease
            lease_seconds: New lease duration
            
        Returns:
            True if extended, False if lease was lost
        """
        ...
    
    @abstractmethod
    async def is_cancelled(self, run_id: UUID) -> bool:
        """
        Check if a run has been cancelled.
        
        Args:
            run_id: Run to check
            
        Returns:
            True if cancelled
        """
        ...
    
    @abstractmethod
    async def cancel(self, run_id: UUID) -> bool:
        """
        Cancel a run.
        
        Args:
            run_id: Run to cancel
            
        Returns:
            True if cancelled, False if not found or already complete
        """
        ...
    
    @abstractmethod
    async def requeue_for_retry(
        self,
        run_id: UUID,
        worker_id: str,
        error: dict,
        delay_seconds: int = 0,
    ) -> bool:
        """
        Requeue a failed run for retry.
        
        Args:
            run_id: Run to retry
            worker_id: Worker releasing the run
            error: Error information
            delay_seconds: Delay before retry
            
        Returns:
            True if requeued, False if max retries exceeded
        """
        ...
    
    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass
