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
    usage: dict = field(default_factory=dict)  # token counts: {prompt_tokens, completion_tokens, total_tokens}
    metadata: dict = field(default_factory=dict)

    # Branching support
    parent_message_id: Optional[UUID] = None  # For branched/edited messages
    branch_id: Optional[UUID] = None  # Groups messages in same branch


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

    # Branching support
    parent_conversation_id: Optional[UUID] = None  # For forked conversations
    active_branch_id: Optional[UUID] = None  # Currently active branch


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

    # Dependencies and scheduling
    dependencies: list[UUID] = field(default_factory=list)  # Task IDs this depends on
    priority: int = 0  # Higher = more important
    due_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Checkpoint for resumable long-running operations
    checkpoint_data: dict = field(default_factory=dict)
    checkpoint_at: Optional[datetime] = None

    # Execution tracking
    attempts: int = 0
    last_error: Optional[str] = None


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


# =============================================================================
# Knowledge Base Models and Store
# =============================================================================


class FactType(str, Enum):
    """Type of fact stored in knowledge base."""

    USER = "user"  # Facts about the user
    PROJECT = "project"  # Facts about the project
    PREFERENCE = "preference"  # Learned preferences
    CONTEXT = "context"  # Contextual information
    CUSTOM = "custom"  # Custom fact type


@dataclass
class Fact:
    """A learned fact about user, project, or context."""

    id: UUID
    key: str  # Unique identifier for the fact
    value: Any  # The fact content
    fact_type: FactType = FactType.CUSTOM
    confidence: float = 1.0  # 0.0 to 1.0
    source: Optional[str] = None  # Where this fact came from

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None  # Optional expiration

    metadata: dict = field(default_factory=dict)


@dataclass
class Summary:
    """A summary of a conversation or set of interactions."""

    id: UUID
    content: str  # The summary text

    # What this summarizes
    conversation_id: Optional[UUID] = None
    conversation_ids: list[UUID] = field(default_factory=list)  # For multi-conversation summaries

    # Time range covered
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class Embedding:
    """
    A vector embedding for semantic search.

    Note: This is optional and requires additional dependencies
    for vector operations (e.g., numpy, faiss, pgvector).
    """

    id: UUID
    vector: list[float]  # The embedding vector

    # What this embedding represents
    content: str  # Original text
    content_type: str = "text"  # text, summary, fact, etc.
    source_id: Optional[UUID] = None  # ID of source object

    model: Optional[str] = None  # Embedding model used
    dimensions: int = 0  # Vector dimensions

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class KnowledgeStore(ABC):
    """
    Abstract interface for knowledge base storage.

    Knowledge stores handle facts, summaries, and optionally
    embeddings for semantic search. This is optional - agents
    can function without a knowledge store.
    """

    # Fact operations
    @abstractmethod
    async def save_fact(self, fact: Fact, scope: Scope = Scope.PROJECT) -> None:
        """Save or update a fact."""
        ...

    @abstractmethod
    async def get_fact(self, fact_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[Fact]:
        """Get a fact by ID."""
        ...

    @abstractmethod
    async def get_fact_by_key(self, key: str, scope: Scope = Scope.PROJECT) -> Optional[Fact]:
        """Get a fact by its key."""
        ...

    @abstractmethod
    async def list_facts(
        self,
        scope: Scope = Scope.PROJECT,
        fact_type: Optional[FactType] = None,
        limit: int = 100,
    ) -> list[Fact]:
        """List facts, optionally filtered by type."""
        ...

    @abstractmethod
    async def delete_fact(self, fact_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        """Delete a fact. Returns True if it existed."""
        ...

    # Summary operations
    @abstractmethod
    async def save_summary(self, summary: Summary, scope: Scope = Scope.PROJECT) -> None:
        """Save or update a summary."""
        ...

    @abstractmethod
    async def get_summary(self, summary_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[Summary]:
        """Get a summary by ID."""
        ...

    @abstractmethod
    async def get_summaries_for_conversation(
        self,
        conversation_id: UUID,
        scope: Scope = Scope.PROJECT,
    ) -> list[Summary]:
        """Get all summaries for a conversation."""
        ...

    @abstractmethod
    async def delete_summary(self, summary_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        """Delete a summary. Returns True if it existed."""
        ...

    # Embedding operations (optional - can raise NotImplementedError)
    async def save_embedding(self, embedding: Embedding, scope: Scope = Scope.PROJECT) -> None:
        """Save an embedding. Optional - may raise NotImplementedError."""
        raise NotImplementedError("Embeddings not supported by this store")

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        scope: Scope = Scope.PROJECT,
        content_type: Optional[str] = None,
    ) -> list[tuple[Embedding, float]]:
        """
        Search for similar embeddings. Returns (embedding, score) tuples.
        Optional - may raise NotImplementedError.
        """
        raise NotImplementedError("Embeddings not supported by this store")

    async def delete_embedding(self, embedding_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        """Delete an embedding. Optional - may raise NotImplementedError."""
        raise NotImplementedError("Embeddings not supported by this store")

    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass


# =============================================================================
# Audit/History Models and Store
# =============================================================================


class AuditEventType(str, Enum):
    """Type of audit event."""

    # Conversation events
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Agent events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # System events
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    CUSTOM = "custom"


class ErrorSeverity(str, Enum):
    """Severity level for errors."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """An audit log entry for tracking interactions."""

    id: UUID
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    conversation_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    agent_key: Optional[str] = None

    # Event details
    action: str = ""  # Human-readable action description
    details: dict = field(default_factory=dict)  # Event-specific data

    # Actor information
    actor_type: str = "agent"  # agent, user, system
    actor_id: Optional[str] = None

    # Request/response tracking
    request_id: Optional[str] = None
    parent_event_id: Optional[UUID] = None  # For nested events

    metadata: dict = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """A record of an error for debugging."""

    id: UUID
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: ErrorSeverity = ErrorSeverity.ERROR

    # Error details
    error_type: str = ""  # Exception class name
    message: str = ""
    stack_trace: Optional[str] = None

    # Context
    conversation_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    agent_key: Optional[str] = None

    # What was happening when error occurred
    context: dict = field(default_factory=dict)

    # Resolution tracking
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

    metadata: dict = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """A performance metric for monitoring."""

    id: UUID
    name: str  # Metric name (e.g., "llm_latency", "tool_execution_time")
    value: float  # Metric value
    unit: str = ""  # Unit of measurement (ms, tokens, etc.)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    conversation_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    agent_key: Optional[str] = None

    # Additional dimensions for grouping/filtering
    tags: dict = field(default_factory=dict)

    metadata: dict = field(default_factory=dict)


class AuditStore(ABC):
    """
    Abstract interface for audit and history storage.

    Audit stores handle interaction logs, error history, and
    performance metrics. This is optional - agents can function
    without an audit store.
    """

    # Audit entry operations
    @abstractmethod
    async def log_event(self, entry: AuditEntry, scope: Scope = Scope.PROJECT) -> None:
        """Log an audit event."""
        ...

    @abstractmethod
    async def get_events(
        self,
        scope: Scope = Scope.PROJECT,
        conversation_id: Optional[UUID] = None,
        run_id: Optional[UUID] = None,
        event_types: Optional[list[AuditEventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit events with optional filters."""
        ...

    # Error operations
    @abstractmethod
    async def log_error(self, error: ErrorRecord, scope: Scope = Scope.PROJECT) -> None:
        """Log an error."""
        ...

    @abstractmethod
    async def get_errors(
        self,
        scope: Scope = Scope.PROJECT,
        severity: Optional[ErrorSeverity] = None,
        resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[ErrorRecord]:
        """Get errors with optional filters."""
        ...

    @abstractmethod
    async def resolve_error(
        self,
        error_id: UUID,
        resolution_notes: Optional[str] = None,
        scope: Scope = Scope.PROJECT,
    ) -> bool:
        """Mark an error as resolved. Returns True if error existed."""
        ...

    # Performance metric operations
    @abstractmethod
    async def record_metric(self, metric: PerformanceMetric, scope: Scope = Scope.PROJECT) -> None:
        """Record a performance metric."""
        ...

    @abstractmethod
    async def get_metrics(
        self,
        name: str,
        scope: Scope = Scope.PROJECT,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[PerformanceMetric]:
        """Get metrics by name with optional filters."""
        ...

    @abstractmethod
    async def get_metric_summary(
        self,
        name: str,
        scope: Scope = Scope.PROJECT,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict:
        """
        Get summary statistics for a metric.
        Returns: {count, min, max, avg, sum, p50, p95, p99}
        """
        ...

    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass
