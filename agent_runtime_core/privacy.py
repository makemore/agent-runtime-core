"""
Privacy and user isolation abstractions for agent_runtime_core.

This module provides framework-agnostic definitions for:

1. **PrivacyConfig** - Settings that control what data is persisted and when.
   Defaults to maximum privacy (auth required for all persistent data).

2. **UserContext** - Represents the current user (authenticated or anonymous).
   Used to scope data access and enforce privacy rules.

3. **MemoryScope** - Defines how memory is scoped (conversation, user, system).

These abstractions are designed to be implemented by framework-specific code
(e.g., django_agent_runtime) while providing consistent privacy guarantees.

Example:
    from agent_runtime_core.privacy import PrivacyConfig, UserContext, MemoryScope

    # Maximum privacy (default)
    config = PrivacyConfig()
    assert config.require_auth_for_memory == True
    assert config.require_auth_for_conversations == True

    # Authenticated user
    user = UserContext(
        user_id="user-123",
        is_authenticated=True,
    )

    # Anonymous user (no persistent data allowed by default)
    anon = UserContext.anonymous()
    assert anon.is_authenticated == False
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MemoryScope(str, Enum):
    """
    Defines how memory is scoped.

    CONVERSATION: Memory is scoped to a single conversation.
                  Lost when conversation ends. Safest option.

    USER: Memory persists across conversations for a user.
          Requires authentication. Good for preferences, learned facts.

    SYSTEM: Memory is shared across all agents in a system for a user.
            Requires authentication. Good for cross-agent knowledge sharing.
    """
    CONVERSATION = "conversation"
    USER = "user"
    SYSTEM = "system"


@dataclass
class PrivacyConfig:
    """
    Configuration for privacy and data persistence.

    All settings default to maximum privacy (auth required for everything).
    Framework implementations should respect these settings.

    Attributes:
        require_auth_for_memory: If True, memory is only stored for authenticated users.
                                 Anonymous users get no persistent memory. Default: True.

        require_auth_for_conversations: If True, conversations are only persisted for
                                        authenticated users. Anonymous conversations
                                        are ephemeral. Default: True.

        require_auth_for_tool_tracking: If True, tool calls are only logged for
                                        authenticated users. Default: True.

        default_memory_scope: Default scope for new memories. Default: CONVERSATION.

        allow_cross_agent_memory: If True, agents in a system can share memories
                                  about a user. Requires SYSTEM scope. Default: True.

        memory_retention_days: How long to retain user memories. None = forever.
                               Default: None.

        conversation_retention_days: How long to retain conversations. None = forever.
                                     Default: None.

        metadata: Additional configuration metadata.

    Example:
        # Maximum privacy (default)
        config = PrivacyConfig()

        # Allow anonymous conversations but not memory
        config = PrivacyConfig(
            require_auth_for_conversations=False,
            require_auth_for_memory=True,
        )

        # Strict retention policy
        config = PrivacyConfig(
            memory_retention_days=30,
            conversation_retention_days=90,
        )
    """
    require_auth_for_memory: bool = True
    require_auth_for_conversations: bool = True
    require_auth_for_tool_tracking: bool = True
    default_memory_scope: MemoryScope = MemoryScope.CONVERSATION
    allow_cross_agent_memory: bool = True
    memory_retention_days: Optional[int] = None
    conversation_retention_days: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def allows_memory(self, user: "UserContext") -> bool:
        """Check if memory storage is allowed for this user."""
        if self.require_auth_for_memory:
            return user.is_authenticated
        return True

    def allows_conversation_persistence(self, user: "UserContext") -> bool:
        """Check if conversation persistence is allowed for this user."""
        if self.require_auth_for_conversations:
            return user.is_authenticated
        return True

    def allows_tool_tracking(self, user: "UserContext") -> bool:
        """Check if tool call tracking is allowed for this user."""
        if self.require_auth_for_tool_tracking:
            return user.is_authenticated
        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "require_auth_for_memory": self.require_auth_for_memory,
            "require_auth_for_conversations": self.require_auth_for_conversations,
            "require_auth_for_tool_tracking": self.require_auth_for_tool_tracking,
            "default_memory_scope": self.default_memory_scope.value,
            "allow_cross_agent_memory": self.allow_cross_agent_memory,
            "memory_retention_days": self.memory_retention_days,
            "conversation_retention_days": self.conversation_retention_days,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrivacyConfig":
        """Deserialize from dictionary."""
        return cls(
            require_auth_for_memory=data.get("require_auth_for_memory", True),
            require_auth_for_conversations=data.get("require_auth_for_conversations", True),
            require_auth_for_tool_tracking=data.get("require_auth_for_tool_tracking", True),
            default_memory_scope=MemoryScope(data.get("default_memory_scope", "conversation")),
            allow_cross_agent_memory=data.get("allow_cross_agent_memory", True),
            memory_retention_days=data.get("memory_retention_days"),
            conversation_retention_days=data.get("conversation_retention_days"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UserContext:
    """
    Represents the current user for privacy and data isolation.

    This is a framework-agnostic representation of a user. Framework
    implementations (e.g., Django) should create UserContext from their
    native user objects.

    Attributes:
        user_id: Unique identifier for the user. None for anonymous users.
        is_authenticated: Whether the user is authenticated.
        username: Optional username for display purposes.
        email: Optional email address.
        metadata: Additional user metadata (roles, permissions, etc.)

    Example:
        # From Django user
        user_ctx = UserContext(
            user_id=str(request.user.id),
            is_authenticated=request.user.is_authenticated,
            username=request.user.username,
            email=request.user.email,
        )

        # Anonymous user
        user_ctx = UserContext.anonymous()

        # Check permissions
        if privacy_config.allows_memory(user_ctx):
            store.save_memory(...)
    """
    user_id: Optional[str] = None
    is_authenticated: bool = False
    username: Optional[str] = None
    email: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> "UserContext":
        """Create an anonymous user context."""
        return cls(
            user_id=None,
            is_authenticated=False,
            username=None,
            email=None,
            metadata={"anonymous": True},
        )

    @classmethod
    def from_user_id(cls, user_id: str, **kwargs) -> "UserContext":
        """Create an authenticated user context from a user ID."""
        return cls(
            user_id=user_id,
            is_authenticated=True,
            **kwargs,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "is_authenticated": self.is_authenticated,
            "username": self.username,
            "email": self.email,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserContext":
        """Deserialize from dictionary."""
        return cls(
            user_id=data.get("user_id"),
            is_authenticated=data.get("is_authenticated", False),
            username=data.get("username"),
            email=data.get("email"),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        if self.is_authenticated:
            return f"User({self.username or self.user_id})"
        return "AnonymousUser"


# Default instances for convenience
DEFAULT_PRIVACY_CONFIG = PrivacyConfig()
ANONYMOUS_USER = UserContext.anonymous()

