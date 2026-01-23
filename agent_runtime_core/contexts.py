"""
Concrete RunContext implementations for different use cases.

These implementations satisfy the RunContext protocol and can be used
directly with StepExecutor and agent runtimes.

Usage:
    # For simple scripts (in-memory, no persistence)
    ctx = InMemoryRunContext(run_id=uuid4())
    
    # For scripts that need persistence across restarts
    ctx = FileRunContext(run_id=uuid4(), checkpoint_dir="./checkpoints")
    
    # Use with StepExecutor
    from agent_runtime_core.steps import StepExecutor, Step
    executor = StepExecutor(ctx)
    results = await executor.run([
        Step("fetch", fetch_data),
        Step("process", process_data),
    ])
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from agent_runtime_core.interfaces import EventType, Message, ToolRegistry


class InMemoryRunContext:
    """
    In-memory RunContext implementation.
    
    Good for:
    - Unit testing
    - Simple scripts that don't need persistence
    - Development and prototyping
    
    State is lost when the process exits.
    
    Example:
        ctx = InMemoryRunContext(
            run_id=uuid4(),
            input_messages=[{"role": "user", "content": "Hello"}],
        )
        
        # Use with an agent
        result = await my_agent.run(ctx)
        
        # Or with StepExecutor
        executor = StepExecutor(ctx)
        results = await executor.run(steps)
    """
    
    def __init__(
        self,
        run_id: Optional[UUID] = None,
        *,
        conversation_id: Optional[UUID] = None,
        input_messages: Optional[list[Message]] = None,
        params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        tool_registry: Optional[ToolRegistry] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        """
        Initialize an in-memory run context.
        
        Args:
            run_id: Unique identifier for this run (auto-generated if not provided)
            conversation_id: Associated conversation ID (optional)
            input_messages: Input messages for this run
            params: Additional parameters
            metadata: Run metadata
            tool_registry: Registry of available tools
            on_event: Optional callback for events (for testing/debugging)
        """
        self._run_id = run_id or uuid4()
        self._conversation_id = conversation_id
        self._input_messages = input_messages or []
        self._params = params or {}
        self._metadata = metadata or {}
        self._tool_registry = tool_registry or ToolRegistry()
        self._cancelled = False
        self._state: Optional[dict] = None
        self._events: list[dict] = []
        self._on_event = on_event
    
    @property
    def run_id(self) -> UUID:
        """Unique identifier for this run."""
        return self._run_id
    
    @property
    def conversation_id(self) -> Optional[UUID]:
        """Conversation this run belongs to (if any)."""
        return self._conversation_id
    
    @property
    def input_messages(self) -> list[Message]:
        """Input messages for this run."""
        return self._input_messages
    
    @property
    def params(self) -> dict:
        """Additional parameters for this run."""
        return self._params
    
    @property
    def metadata(self) -> dict:
        """Metadata associated with this run."""
        return self._metadata
    
    @property
    def tool_registry(self) -> ToolRegistry:
        """Registry of available tools for this agent."""
        return self._tool_registry
    
    async def emit(self, event_type: EventType | str, payload: dict) -> None:
        """Emit an event (stored in memory)."""
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        event = {
            "event_type": event_type_str,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._events.append(event)
        if self._on_event:
            self._on_event(event_type_str, payload)
    
    async def checkpoint(self, state: dict) -> None:
        """Save a state checkpoint (in memory)."""
        self._state = state
    
    async def get_state(self) -> Optional[dict]:
        """Get the last checkpointed state."""
        return self._state
    
    def cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled
    
    def cancel(self) -> None:
        """Request cancellation of this run."""
        self._cancelled = True
    
    @property
    def events(self) -> list[dict]:
        """Get all emitted events (for testing/debugging)."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear all events (for testing)."""
        self._events.clear()


class FileRunContext:
    """
    File-based RunContext implementation with persistent checkpoints.

    Good for:
    - Scripts that need to resume after restart
    - Long-running processes without a database
    - Simple persistence without external dependencies

    Checkpoints are saved as JSON files in the specified directory.

    Example:
        ctx = FileRunContext(
            run_id=uuid4(),
            checkpoint_dir="./checkpoints",
            input_messages=[{"role": "user", "content": "Process this"}],
        )

        # Checkpoints are saved to ./checkpoints/{run_id}.json
        executor = StepExecutor(ctx)
        results = await executor.run(steps)

        # To resume after restart, use the same run_id:
        ctx = FileRunContext(run_id=previous_run_id, checkpoint_dir="./checkpoints")
        results = await executor.run(steps, resume=True)
    """

    def __init__(
        self,
        run_id: Optional[UUID] = None,
        *,
        checkpoint_dir: str = "./checkpoints",
        conversation_id: Optional[UUID] = None,
        input_messages: Optional[list[Message]] = None,
        params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        tool_registry: Optional[ToolRegistry] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        """
        Initialize a file-based run context.

        Args:
            run_id: Unique identifier for this run (auto-generated if not provided)
            checkpoint_dir: Directory to store checkpoint files
            conversation_id: Associated conversation ID (optional)
            input_messages: Input messages for this run
            params: Additional parameters
            metadata: Run metadata
            tool_registry: Registry of available tools
            on_event: Optional callback for events
        """
        self._run_id = run_id or uuid4()
        self._checkpoint_dir = Path(checkpoint_dir)
        self._conversation_id = conversation_id
        self._input_messages = input_messages or []
        self._params = params or {}
        self._metadata = metadata or {}
        self._tool_registry = tool_registry or ToolRegistry()
        self._cancelled = False
        self._on_event = on_event
        self._state_cache: Optional[dict] = None

        # Ensure checkpoint directory exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> UUID:
        """Unique identifier for this run."""
        return self._run_id

    @property
    def conversation_id(self) -> Optional[UUID]:
        """Conversation this run belongs to (if any)."""
        return self._conversation_id

    @property
    def input_messages(self) -> list[Message]:
        """Input messages for this run."""
        return self._input_messages

    @property
    def params(self) -> dict:
        """Additional parameters for this run."""
        return self._params

    @property
    def metadata(self) -> dict:
        """Metadata associated with this run."""
        return self._metadata

    @property
    def tool_registry(self) -> ToolRegistry:
        """Registry of available tools for this agent."""
        return self._tool_registry

    def _checkpoint_path(self) -> Path:
        """Get the path to the checkpoint file for this run."""
        return self._checkpoint_dir / f"{self._run_id}.json"

    def _events_path(self) -> Path:
        """Get the path to the events file for this run."""
        return self._checkpoint_dir / f"{self._run_id}_events.jsonl"

    async def emit(self, event_type: EventType | str, payload: dict) -> None:
        """Emit an event (appended to events file)."""
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        event = {
            "event_type": event_type_str,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Append to events file (JSONL format)
        with open(self._events_path(), "a") as f:
            f.write(json.dumps(event) + "\n")

        if self._on_event:
            self._on_event(event_type_str, payload)

    async def checkpoint(self, state: dict) -> None:
        """Save a state checkpoint to file."""
        self._state_cache = state
        checkpoint_data = {
            "run_id": str(self._run_id),
            "state": state,
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Write atomically using temp file
        temp_path = self._checkpoint_path().with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        temp_path.rename(self._checkpoint_path())

    async def get_state(self) -> Optional[dict]:
        """Get the last checkpointed state from file."""
        if self._state_cache is not None:
            return self._state_cache

        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            self._state_cache = data.get("state")
            return self._state_cache
        except (json.JSONDecodeError, IOError):
            return None

    def cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation of this run."""
        self._cancelled = True

    def get_events(self) -> list[dict]:
        """Read all events from the events file."""
        events_path = self._events_path()
        if not events_path.exists():
            return []

        events = []
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

    def clear(self) -> None:
        """Delete checkpoint and events files for this run."""
        checkpoint_path = self._checkpoint_path()
        events_path = self._events_path()

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if events_path.exists():
            events_path.unlink()

        self._state_cache = None

