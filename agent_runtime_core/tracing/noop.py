"""
No-op trace sink implementation.

Used when tracing is disabled or not configured.
"""

from typing import Optional
from uuid import UUID

from agent_runtime_core.interfaces import TraceSink


class NoopTraceSink(TraceSink):
    """
    No-op trace sink that discards all traces.
    
    Used when tracing is disabled.
    """
    
    def start_run(self, run_id: UUID, metadata: dict) -> None:
        """No-op."""
        pass
    
    def log_event(self, run_id: UUID, event_type: str, payload: dict) -> None:
        """No-op."""
        pass
    
    def end_run(self, run_id: UUID, outcome: str, metadata: Optional[dict] = None) -> None:
        """No-op."""
        pass
    
    def flush(self) -> None:
        """No-op."""
        pass
