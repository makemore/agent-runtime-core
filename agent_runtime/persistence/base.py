"""
Abstract base classes for persistence backends.

These interfaces define the contract that all storage backends must implement.
Projects depending on agent-runtime-core can provide their own implementations
(e.g., database-backed, cloud storage, etc.).

For Django/database implementations:
- The `scope` parameter can be ignored if you use user/tenant context instead
- Store implementations receive context through their constructor (e.g., user, org)
- The abstract methods still accept scope for interface compatibility, but
  implementations can choose to ignore it

Example Django implementation:
    class DjangoMemoryStore(MemoryStore):
        def __init__(self, user):
            self.user = user

        async def get(self, key: str, scope: Scope = Scope.PROJECT) -> Optional[Any]:
            # Ignore scope, use self.user instead
            try:
                entry = await Memory.objects.aget(user=self.user, key=key)
                return entry.value
            except Memory.DoesNotExist:
                return None
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, AsyncIterator
from uuid import UUID


class Scope(str, Enum):
    """
    Storage scope for memory and other persistent data.

    For file-based storage:
    - GLOBAL: User's home directory (~/.agent_runtime/)
    - PROJECT: Current working directory (./.agent_runtime/)
    - SESSION: In-memory only, not persisted

    For database-backed storage, implementations may ignore this
    and use user/tenant context from the store constructor instead.
    """

    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"


class TaskState(str, Enum):
    """State of a task."""
    
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass
class ToolCall:
    """A tool call made during a conversation."""
    
    id: str
    name: str
    arguments: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolResult:
    """Result of a tool call."""
    
    tool_call_id: str
    result: Any
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationMessage:
    """A message in a conversation with full state."""
    
    id: UUID
    role: str  # system, user, assistant, tool
    content: str | dict | list
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # For assistant messages with tool calls
    tool_calls: list[ToolCall] = field(default_factory=list)
    
    # For tool result messages
    tool_call_id: Optional[str] = None
    
    # Metadata
    model: Optional[str] = None
    usage: dict = field(default_factory=dict)  # token counts
    metadata: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """A complete conversation with all state."""
    
    id: UUID
    title: Optional[str] = None
    messages: list[ConversationMessage] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    # Associated agent
    agent_key: Optional[str] = None
    
    # Summary for long conversations
    summary: Optional[str] = None


@dataclass
class Task:
    """A task in a task list."""
    
    id: UUID
    name: str
    description: str = ""
    state: TaskState = TaskState.NOT_STARTED
    parent_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskList:
    """A list of tasks."""
    
    id: UUID
    name: str
    tasks: list[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Associated conversation/run
    conversation_id: Optional[UUID] = None
    run_id: Optional[UUID] = None


class MemoryStore(ABC):
    """
    Abstract interface for key-value memory storage.
    
    Memory stores handle persistent key-value data that agents can
    use to remember information across sessions.
    """
    
    @abstractmethod
    async def get(self, key: str, scope: Scope = Scope.PROJECT) -> Optional[Any]:
        """Get a value by key."""
        ...
    
    @abstractmethod
    async def set(self, key: str, value: Any, scope: Scope = Scope.PROJECT) -> None:
        """Set a value by key."""
        ...
    
    @abstractmethod
    async def delete(self, key: str, scope: Scope = Scope.PROJECT) -> bool:
        """Delete a key. Returns True if key existed."""
        ...
    
    @abstractmethod
    async def list_keys(self, scope: Scope = Scope.PROJECT, prefix: Optional[str] = None) -> list[str]:
        """List all keys, optionally filtered by prefix."""
        ...
    
    @abstractmethod
    async def clear(self, scope: Scope = Scope.PROJECT) -> None:
        """Clear all keys in the given scope."""
        ...
    
    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass


class ConversationStore(ABC):
    """
    Abstract interface for conversation history storage.

    Conversation stores handle full conversation state including
    messages, tool calls, and metadata.
    """

    @abstractmethod
    async def save(self, conversation: Conversation, scope: Scope = Scope.PROJECT) -> None:
        """Save or update a conversation."""
        ...

    @abstractmethod
    async def get(self, conversation_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[Conversation]:
        """Get a conversation by ID."""
        ...

    @abstractmethod
    async def delete(self, conversation_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        """Delete a conversation. Returns True if it existed."""
        ...

    @abstractmethod
    async def list_conversations(
        self,
        scope: Scope = Scope.PROJECT,
        limit: int = 100,
        offset: int = 0,
        agent_key: Optional[str] = None,
    ) -> list[Conversation]:
        """List conversations, optionally filtered by agent."""
        ...

    @abstractmethod
    async def add_message(
        self,
        conversation_id: UUID,
        message: ConversationMessage,
        scope: Scope = Scope.PROJECT,
    ) -> None:
        """Add a message to an existing conversation."""
        ...

    @abstractmethod
    async def get_messages(
        self,
        conversation_id: UUID,
        scope: Scope = Scope.PROJECT,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> list[ConversationMessage]:
        """Get messages from a conversation."""
        ...

    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass


class TaskStore(ABC):
    """
    Abstract interface for task list storage.

    Task stores handle task lists and their state for tracking
    agent progress on complex work.
    """

    @abstractmethod
    async def save(self, task_list: TaskList, scope: Scope = Scope.PROJECT) -> None:
        """Save or update a task list."""
        ...

    @abstractmethod
    async def get(self, task_list_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[TaskList]:
        """Get a task list by ID."""
        ...

    @abstractmethod
    async def delete(self, task_list_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        """Delete a task list. Returns True if it existed."""
        ...

    @abstractmethod
    async def get_by_conversation(
        self,
        conversation_id: UUID,
        scope: Scope = Scope.PROJECT,
    ) -> Optional[TaskList]:
        """Get the task list associated with a conversation."""
        ...

    @abstractmethod
    async def update_task(
        self,
        task_list_id: UUID,
        task_id: UUID,
        state: Optional[TaskState] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Scope = Scope.PROJECT,
    ) -> None:
        """Update a specific task in a task list."""
        ...

    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass


class PreferencesStore(ABC):
    """
    Abstract interface for preferences storage.

    Preferences stores handle user and agent configuration
    that persists across sessions.
    """

    @abstractmethod
    async def get(self, key: str, scope: Scope = Scope.GLOBAL) -> Optional[Any]:
        """Get a preference value."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, scope: Scope = Scope.GLOBAL) -> None:
        """Set a preference value."""
        ...

    @abstractmethod
    async def delete(self, key: str, scope: Scope = Scope.GLOBAL) -> bool:
        """Delete a preference. Returns True if it existed."""
        ...

    @abstractmethod
    async def get_all(self, scope: Scope = Scope.GLOBAL) -> dict[str, Any]:
        """Get all preferences in the given scope."""
        ...

    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass

