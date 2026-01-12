"""
In-memory state store implementation.

Good for:
- Unit testing
- Local development
- Simple single-process scripts
"""

from typing import Optional
from uuid import UUID

from agent_runtime_core.state.base import StateStore


class InMemoryStateStore(StateStore):
    """
    In-memory state store implementation.
    
    Stores state in memory. Data is lost when the process exits.
    """
    
    def __init__(self):
        # run_id -> list of checkpoints (ordered by time)
        self._checkpoints: dict[UUID, list[dict]] = {}
        # run_id -> status
        self._statuses: dict[UUID, str] = {}
    
    async def save_checkpoint(self, run_id: UUID, state: dict) -> None:
        """Save a checkpoint for a run."""
        if run_id not in self._checkpoints:
            self._checkpoints[run_id] = []
        self._checkpoints[run_id].append(state)
    
    async def get_checkpoint(self, run_id: UUID) -> Optional[dict]:
        """Get the latest checkpoint for a run."""
        checkpoints = self._checkpoints.get(run_id, [])
        return checkpoints[-1] if checkpoints else None
    
    async def update_run_status(self, run_id: UUID, status: str) -> None:
        """Update the status of a run."""
        self._statuses[run_id] = status
    
    async def get_run_status(self, run_id: UUID) -> Optional[str]:
        """Get the status of a run."""
        return self._statuses.get(run_id)
    
    def clear(self) -> None:
        """Clear all state. Useful for testing."""
        self._checkpoints.clear()
        self._statuses.clear()
