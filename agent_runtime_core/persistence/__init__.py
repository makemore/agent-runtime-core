"""
Persistence module for agent state, memory, and conversation history.

This module provides pluggable storage backends for:
- Memory (global and project-scoped key-value storage)
- Conversation history (full conversation state including tool calls)
- Task state (task lists and progress)
- Preferences (user and agent configuration)

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
    MemoryStore,
    ConversationStore,
    TaskStore,
    PreferencesStore,
    Scope,
    Conversation,
    ConversationMessage,
    ToolCall,
    ToolResult,
    TaskList,
    Task,
    TaskState,
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
    # Abstract interfaces
    "MemoryStore",
    "ConversationStore",
    "TaskStore",
    "PreferencesStore",
    "Scope",
    # Data classes
    "Conversation",
    "ConversationMessage",
    "ToolCall",
    "ToolResult",
    "TaskList",
    "Task",
    "TaskState",
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

