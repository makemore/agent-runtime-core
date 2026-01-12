"""
Abstract base class for state store implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID


class StateStore(ABC):
    """
    Abstract interface for state storage.
    
    State stores handle:
    - Run state (status, metadata)
    - Checkpoints for recovery
    """
    
    @abstractmethod
    async def save_checkpoint(self, run_id: UUID, state: dict) -> None:
        """
        Save a checkpoint for a run.
        
        Args:
            run_id: Run identifier
            state: State to checkpoint
        """
        ...
    
    @abstractmethod
    async def get_checkpoint(self, run_id: UUID) -> Optional[dict]:
        """
        Get the latest checkpoint for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Latest checkpoint state, or None if no checkpoint exists
        """
        ...
    
    @abstractmethod
    async def update_run_status(self, run_id: UUID, status: str) -> None:
        """
        Update the status of a run.
        
        Args:
            run_id: Run identifier
            status: New status
        """
        ...
    
    @abstractmethod
    async def get_run_status(self, run_id: UUID) -> Optional[str]:
        """
        Get the status of a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run status, or None if run not found
        """
        ...
    
    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass
