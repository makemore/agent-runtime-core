"""
Persistence manager for coordinating storage backends.

The PersistenceManager provides a unified interface for accessing
all persistence stores with configurable backends.

For Django integration, you can either:
1. Pass pre-instantiated store instances
2. Pass store classes with appropriate kwargs
3. Use factory functions for request-scoped stores
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

from agent_runtime.persistence.base import (
    MemoryStore,
    ConversationStore,
    TaskStore,
    PreferencesStore,
    Scope,
)
from agent_runtime.persistence.file import (
    FileMemoryStore,
    FileConversationStore,
    FileTaskStore,
    FilePreferencesStore,
)


# Type aliases for store factories
MemoryStoreFactory = Callable[[], MemoryStore]
ConversationStoreFactory = Callable[[], ConversationStore]
TaskStoreFactory = Callable[[], TaskStore]
PreferencesStoreFactory = Callable[[], PreferencesStore]


@dataclass
class PersistenceConfig:
    """
    Configuration for persistence backends.

    Each store can be configured as:
    - A class (will be instantiated with store_kwargs)
    - A pre-instantiated store instance
    - A factory function that returns a store instance

    Example for Django:
        from myapp.stores import DjangoMemoryStore, DjangoConversationStore

        # Option 1: Pass classes with kwargs
        config = PersistenceConfig(
            memory_store_class=DjangoMemoryStore,
            memory_store_kwargs={"user": request.user},
        )

        # Option 2: Pass pre-instantiated stores
        config = PersistenceConfig(
            memory_store=DjangoMemoryStore(user=request.user),
            conversation_store=DjangoConversationStore(user=request.user),
        )

        # Option 3: Pass factory functions
        config = PersistenceConfig(
            memory_store_factory=lambda: DjangoMemoryStore(user=get_current_user()),
        )
    """

    # Backend classes (can be swapped for custom implementations)
    memory_store_class: Type[MemoryStore] = FileMemoryStore
    conversation_store_class: Type[ConversationStore] = FileConversationStore
    task_store_class: Type[TaskStore] = FileTaskStore
    preferences_store_class: Type[PreferencesStore] = FilePreferencesStore

    # Pre-instantiated store instances (takes precedence over classes)
    memory_store: Optional[MemoryStore] = None
    conversation_store: Optional[ConversationStore] = None
    task_store: Optional[TaskStore] = None
    preferences_store: Optional[PreferencesStore] = None

    # Factory functions (takes precedence over classes, but not instances)
    memory_store_factory: Optional[MemoryStoreFactory] = None
    conversation_store_factory: Optional[ConversationStoreFactory] = None
    task_store_factory: Optional[TaskStoreFactory] = None
    preferences_store_factory: Optional[PreferencesStoreFactory] = None

    # Kwargs passed to store class constructors (only used with classes)
    memory_store_kwargs: dict = field(default_factory=dict)
    conversation_store_kwargs: dict = field(default_factory=dict)
    task_store_kwargs: dict = field(default_factory=dict)
    preferences_store_kwargs: dict = field(default_factory=dict)

    # Project directory (convenience for file-based stores)
    # Only used if store_kwargs doesn't already have project_dir
    project_dir: Optional[Path] = None


class PersistenceManager:
    """
    Unified manager for all persistence stores.

    Provides access to memory, conversations, tasks, and preferences
    with pluggable backends.

    Example:
        # Use default file-based storage
        manager = PersistenceManager()

        # Store global memory
        await manager.memory.set("user_name", "Alice", scope=Scope.GLOBAL)

        # Store project-specific memory
        await manager.memory.set("project_type", "python", scope=Scope.PROJECT)

        # Save a conversation
        await manager.conversations.save(conversation)

        # Use custom backends (Django example)
        config = PersistenceConfig(
            memory_store=DjangoMemoryStore(user=request.user),
            conversation_store=DjangoConversationStore(user=request.user),
        )
        manager = PersistenceManager(config)
    """

    def __init__(self, config: Optional[PersistenceConfig] = None):
        self._config = config or PersistenceConfig()
        self._memory: Optional[MemoryStore] = None
        self._conversations: Optional[ConversationStore] = None
        self._tasks: Optional[TaskStore] = None
        self._preferences: Optional[PreferencesStore] = None

    def _build_kwargs(self, store_kwargs: dict) -> dict:
        """Build kwargs for store instantiation."""
        kwargs = {}
        # Only add project_dir if not already in store_kwargs
        if self._config.project_dir and "project_dir" not in store_kwargs:
            kwargs["project_dir"] = self._config.project_dir
        kwargs.update(store_kwargs)
        return kwargs

    @property
    def memory(self) -> MemoryStore:
        """Get the memory store."""
        if self._memory is None:
            # Priority: instance > factory > class
            if self._config.memory_store is not None:
                self._memory = self._config.memory_store
            elif self._config.memory_store_factory is not None:
                self._memory = self._config.memory_store_factory()
            else:
                kwargs = self._build_kwargs(self._config.memory_store_kwargs)
                self._memory = self._config.memory_store_class(**kwargs)
        return self._memory

    @property
    def conversations(self) -> ConversationStore:
        """Get the conversation store."""
        if self._conversations is None:
            if self._config.conversation_store is not None:
                self._conversations = self._config.conversation_store
            elif self._config.conversation_store_factory is not None:
                self._conversations = self._config.conversation_store_factory()
            else:
                kwargs = self._build_kwargs(self._config.conversation_store_kwargs)
                self._conversations = self._config.conversation_store_class(**kwargs)
        return self._conversations

    @property
    def tasks(self) -> TaskStore:
        """Get the task store."""
        if self._tasks is None:
            if self._config.task_store is not None:
                self._tasks = self._config.task_store
            elif self._config.task_store_factory is not None:
                self._tasks = self._config.task_store_factory()
            else:
                kwargs = self._build_kwargs(self._config.task_store_kwargs)
                self._tasks = self._config.task_store_class(**kwargs)
        return self._tasks

    @property
    def preferences(self) -> PreferencesStore:
        """Get the preferences store."""
        if self._preferences is None:
            if self._config.preferences_store is not None:
                self._preferences = self._config.preferences_store
            elif self._config.preferences_store_factory is not None:
                self._preferences = self._config.preferences_store_factory()
            else:
                kwargs = self._build_kwargs(self._config.preferences_store_kwargs)
                self._preferences = self._config.preferences_store_class(**kwargs)
        return self._preferences

    async def close(self) -> None:
        """Close all stores."""
        if self._memory:
            await self._memory.close()
        if self._conversations:
            await self._conversations.close()
        if self._tasks:
            await self._tasks.close()
        if self._preferences:
            await self._preferences.close()


# Global manager instance
_manager: Optional[PersistenceManager] = None
_config: Optional[PersistenceConfig] = None


def configure_persistence(
    memory_store_class: Optional[Type[MemoryStore]] = None,
    conversation_store_class: Optional[Type[ConversationStore]] = None,
    task_store_class: Optional[Type[TaskStore]] = None,
    preferences_store_class: Optional[Type[PreferencesStore]] = None,
    project_dir: Optional[Path] = None,
    **kwargs,
) -> PersistenceConfig:
    """
    Configure the global persistence manager.
    
    Args:
        memory_store_class: Custom memory store implementation
        conversation_store_class: Custom conversation store implementation
        task_store_class: Custom task store implementation
        preferences_store_class: Custom preferences store implementation
        project_dir: Project directory for PROJECT scope
        **kwargs: Additional store-specific configuration
    
    Returns:
        The configured PersistenceConfig
    """
    global _config, _manager
    
    config = PersistenceConfig(project_dir=project_dir)
    
    if memory_store_class:
        config.memory_store_class = memory_store_class
    if conversation_store_class:
        config.conversation_store_class = conversation_store_class
    if task_store_class:
        config.task_store_class = task_store_class
    if preferences_store_class:
        config.preferences_store_class = preferences_store_class
    
    _config = config
    _manager = None  # Reset manager to use new config
    
    return config


def get_persistence_manager() -> PersistenceManager:
    """
    Get the global persistence manager.
    
    Creates a new manager with default config if not configured.
    """
    global _manager
    
    if _manager is None:
        _manager = PersistenceManager(_config)
    
    return _manager

