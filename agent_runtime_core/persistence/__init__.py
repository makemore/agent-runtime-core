"""
Persistence module for agent state, memory, and conversation history.

This module provides pluggable storage backends for:
- Memory (global and project-scoped key-value storage)
- Conversation history (full conversation state including tool calls)
- Task state (task lists and progress)
- Preferences (user and agent configuration)
- Knowledge base (facts, summaries, embeddings) - optional
- Audit/history (logs, errors, metrics) - optional

Example usage:
    from agent_runtime_core.persistence import (
        MemoryStore,
        ConversationStore,
        FileMemoryStore,
        FileConversationStore,
        PersistenceManager,
        Scope,
    )

    # Use the high-level manager
    manager = PersistenceManager()

    # Store global memory
    await manager.memory.set("user_name", "Alice", scope=Scope.GLOBAL)

    # Store project-specific memory
    await manager.memory.set("project_type", "python", scope=Scope.PROJECT)

    # Save a conversation
    await manager.conversations.save(conversation)
"""

from agent_runtime_core.persistence.base import (
    # Core stores
    MemoryStore,
    ConversationStore,
    TaskStore,
    PreferencesStore,
    # Optional stores
    KnowledgeStore,
    AuditStore,
    # Enums
    Scope,
    TaskState,
    FactType,
    AuditEventType,
    ErrorSeverity,
    # Conversation models
    Conversation,
    ConversationMessage,
    ToolCall,
    ToolResult,
    # Task models
    TaskList,
    Task,
    # Knowledge models
    Fact,
    Summary,
    Embedding,
    # Audit models
    AuditEntry,
    ErrorRecord,
    PerformanceMetric,
)

from agent_runtime_core.persistence.file import (
    FileMemoryStore,
    FileConversationStore,
    FileTaskStore,
    FilePreferencesStore,
)

from agent_runtime_core.persistence.manager import (
    PersistenceManager,
    PersistenceConfig,
    get_persistence_manager,
    configure_persistence,
)

__all__ = [
    # Abstract interfaces - core
    "MemoryStore",
    "ConversationStore",
    "TaskStore",
    "PreferencesStore",
    # Abstract interfaces - optional
    "KnowledgeStore",
    "AuditStore",
    # Enums
    "Scope",
    "TaskState",
    "FactType",
    "AuditEventType",
    "ErrorSeverity",
    # Conversation models
    "Conversation",
    "ConversationMessage",
    "ToolCall",
    "ToolResult",
    # Task models
    "TaskList",
    "Task",
    # Knowledge models
    "Fact",
    "Summary",
    "Embedding",
    # Audit models
    "AuditEntry",
    "ErrorRecord",
    "PerformanceMetric",
    # File implementations
    "FileMemoryStore",
    "FileConversationStore",
    "FileTaskStore",
    "FilePreferencesStore",
    # Manager
    "PersistenceManager",
    "PersistenceConfig",
    "get_persistence_manager",
    "configure_persistence",
]

