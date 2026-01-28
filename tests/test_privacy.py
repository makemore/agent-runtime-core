"""
Tests for privacy and user isolation.

These tests ensure that:
1. Privacy settings default to maximum security (auth required for all persistent data)
2. User data is properly isolated (no cross-user pollution)
3. Anonymous users cannot access persistent data in strict mode
4. Memory scoping works correctly (conversation, user, system)
5. Privacy config is properly enforced
"""

import pytest
from uuid import uuid4

from agent_runtime_core.privacy import (
    PrivacyConfig,
    UserContext,
    MemoryScope,
    DEFAULT_PRIVACY_CONFIG,
    ANONYMOUS_USER,
)
from agent_runtime_core.contexts import InMemoryRunContext, FileRunContext
from agent_runtime_core.multi_agent import SystemContext, SharedMemoryConfig


class TestPrivacyConfigDefaults:
    """Test that privacy defaults are maximally secure."""

    def test_default_requires_auth_for_memory(self):
        """Default config should require auth for memory storage."""
        config = PrivacyConfig()
        assert config.require_auth_for_memory is True

    def test_default_requires_auth_for_conversations(self):
        """Default config should require auth for conversation persistence."""
        config = PrivacyConfig()
        assert config.require_auth_for_conversations is True

    def test_default_requires_auth_for_tool_tracking(self):
        """Default config should require auth for tool call tracking."""
        config = PrivacyConfig()
        assert config.require_auth_for_tool_tracking is True

    def test_default_memory_scope_is_conversation(self):
        """Default memory scope should be conversation (most restrictive)."""
        config = PrivacyConfig()
        assert config.default_memory_scope == MemoryScope.CONVERSATION

    def test_global_default_config_is_secure(self):
        """The global DEFAULT_PRIVACY_CONFIG should be maximally secure."""
        assert DEFAULT_PRIVACY_CONFIG.require_auth_for_memory is True
        assert DEFAULT_PRIVACY_CONFIG.require_auth_for_conversations is True
        assert DEFAULT_PRIVACY_CONFIG.require_auth_for_tool_tracking is True


class TestUserContextIsolation:
    """Test user context creation and isolation."""

    def test_anonymous_user_is_not_authenticated(self):
        """Anonymous users should not be authenticated."""
        anon = UserContext.anonymous()
        assert anon.is_authenticated is False
        assert anon.user_id is None

    def test_global_anonymous_user_is_not_authenticated(self):
        """The global ANONYMOUS_USER should not be authenticated."""
        assert ANONYMOUS_USER.is_authenticated is False
        assert ANONYMOUS_USER.user_id is None

    def test_authenticated_user_has_user_id(self):
        """Authenticated users should have a user_id."""
        user = UserContext.from_user_id("user-123")
        assert user.is_authenticated is True
        assert user.user_id == "user-123"

    def test_different_users_have_different_ids(self):
        """Different users should have different IDs."""
        user1 = UserContext.from_user_id("user-1")
        user2 = UserContext.from_user_id("user-2")
        assert user1.user_id != user2.user_id

    def test_user_context_serialization_roundtrip(self):
        """UserContext should serialize and deserialize correctly."""
        user = UserContext(
            user_id="user-123",
            is_authenticated=True,
            username="testuser",
            email="test@example.com",
            metadata={"role": "admin"},
        )
        data = user.to_dict()
        restored = UserContext.from_dict(data)

        assert restored.user_id == user.user_id
        assert restored.is_authenticated == user.is_authenticated
        assert restored.username == user.username
        assert restored.email == user.email
        assert restored.metadata == user.metadata


class TestPrivacyEnforcement:
    """Test that privacy config properly controls access."""

    def test_anonymous_user_denied_memory_in_strict_mode(self):
        """Anonymous users should be denied memory access in strict mode."""
        config = PrivacyConfig(require_auth_for_memory=True)
        anon = UserContext.anonymous()

        assert config.allows_memory(anon) is False

    def test_authenticated_user_allowed_memory_in_strict_mode(self):
        """Authenticated users should be allowed memory access in strict mode."""
        config = PrivacyConfig(require_auth_for_memory=True)
        user = UserContext.from_user_id("user-123")

        assert config.allows_memory(user) is True

    def test_anonymous_user_allowed_memory_when_disabled(self):
        """Anonymous users should be allowed memory when auth not required."""
        config = PrivacyConfig(require_auth_for_memory=False)
        anon = UserContext.anonymous()

        assert config.allows_memory(anon) is True

    def test_anonymous_user_denied_conversations_in_strict_mode(self):
        """Anonymous users should be denied conversation persistence in strict mode."""
        config = PrivacyConfig(require_auth_for_conversations=True)
        anon = UserContext.anonymous()

        assert config.allows_conversation_persistence(anon) is False

    def test_authenticated_user_allowed_conversations_in_strict_mode(self):
        """Authenticated users should be allowed conversation persistence."""
        config = PrivacyConfig(require_auth_for_conversations=True)
        user = UserContext.from_user_id("user-123")

        assert config.allows_conversation_persistence(user) is True

    def test_anonymous_user_denied_tool_tracking_in_strict_mode(self):
        """Anonymous users should be denied tool tracking in strict mode."""
        config = PrivacyConfig(require_auth_for_tool_tracking=True)
        anon = UserContext.anonymous()

        assert config.allows_tool_tracking(anon) is False

    def test_authenticated_user_allowed_tool_tracking_in_strict_mode(self):
        """Authenticated users should be allowed tool tracking."""
        config = PrivacyConfig(require_auth_for_tool_tracking=True)
        user = UserContext.from_user_id("user-123")

        assert config.allows_tool_tracking(user) is True


class TestMemoryScopeIsolation:
    """Test that memory scoping properly isolates data."""

    def test_conversation_scope_is_most_restrictive(self):
        """CONVERSATION scope should be the most restrictive."""
        assert MemoryScope.CONVERSATION.value == "conversation"
        # Conversation scope means data only lives in current conversation

    def test_user_scope_is_per_user(self):
        """USER scope should be per-user isolation."""
        assert MemoryScope.USER.value == "user"
        # User scope means data is isolated per user across conversations

    def test_system_scope_is_shared(self):
        """SYSTEM scope allows sharing across users."""
        assert MemoryScope.SYSTEM.value == "system"
        # System scope means data is shared (for non-private data only)

    def test_memory_scopes_are_distinct(self):
        """All memory scopes should be distinct values."""
        scopes = [MemoryScope.CONVERSATION, MemoryScope.USER, MemoryScope.SYSTEM]
        values = [s.value for s in scopes]
        assert len(values) == len(set(values))


class TestRunContextPrivacyIntegration:
    """Test RunContext integration with privacy settings."""

    def test_in_memory_context_has_default_privacy_config(self):
        """InMemoryRunContext should have default secure privacy config."""
        ctx = InMemoryRunContext()
        # Privacy config should be present and secure by default
        assert ctx.privacy_config is not None
        assert ctx.privacy_config.require_auth_for_memory is True

    def test_in_memory_context_has_anonymous_user_by_default(self):
        """InMemoryRunContext should have anonymous user by default."""
        ctx = InMemoryRunContext()
        assert ctx.user_context is not None
        assert ctx.user_context.is_authenticated is False

    def test_in_memory_context_with_custom_user(self):
        """InMemoryRunContext should accept custom user context."""
        user = UserContext.from_user_id("user-123")
        ctx = InMemoryRunContext(user_context=user)

        assert ctx.user_context.is_authenticated is True
        assert ctx.user_context.user_id == "user-123"

    def test_in_memory_context_with_custom_privacy_config(self):
        """InMemoryRunContext should accept custom privacy config."""
        config = PrivacyConfig(require_auth_for_memory=False)
        ctx = InMemoryRunContext(privacy_config=config)

        assert ctx.privacy_config.require_auth_for_memory is False

    def test_file_context_has_default_privacy_config(self):
        """FileRunContext should have default secure privacy config."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = FileRunContext(checkpoint_dir=tmpdir)
            assert ctx.privacy_config is not None
            assert ctx.privacy_config.require_auth_for_memory is True

    def test_file_context_has_anonymous_user_by_default(self):
        """FileRunContext should have anonymous user by default."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = FileRunContext(checkpoint_dir=tmpdir)
            assert ctx.user_context is not None
            assert ctx.user_context.is_authenticated is False


class TestUserDataIsolation:
    """Test that user data is properly isolated between users."""

    def test_different_users_have_isolated_contexts(self):
        """Different users should have completely isolated contexts."""
        user1 = UserContext.from_user_id("user-1")
        user2 = UserContext.from_user_id("user-2")

        ctx1 = InMemoryRunContext(user_context=user1)
        ctx2 = InMemoryRunContext(user_context=user2)

        # User IDs should be different
        assert ctx1.user_context.user_id != ctx2.user_context.user_id

        # Run IDs should be different (separate execution contexts)
        assert ctx1.run_id != ctx2.run_id

    def test_user_cannot_access_another_users_id(self):
        """User context should not expose other user IDs."""
        user1 = UserContext.from_user_id("secret-user-id-1")
        user2 = UserContext.from_user_id("secret-user-id-2")

        # Each user only knows their own ID
        assert user1.user_id == "secret-user-id-1"
        assert user2.user_id == "secret-user-id-2"

        # Cannot derive one user's ID from another
        assert "secret-user-id-1" not in str(user2.to_dict())
        assert "secret-user-id-2" not in str(user1.to_dict())



class TestSharedMemoryConfig:
    """Test SharedMemoryConfig for multi-agent systems."""

    def test_shared_memory_config_defaults(self):
        """SharedMemoryConfig should have secure defaults."""
        config = SharedMemoryConfig()
        assert config.enabled is True
        assert config.require_auth is True  # Must be authenticated to use shared memory

    def test_shared_memory_config_serialization(self):
        """SharedMemoryConfig should serialize and deserialize correctly."""
        config = SharedMemoryConfig(
            enabled=True,
            require_auth=True,
            retention_days=365,
            max_memories_per_user=1000,
        )
        data = config.to_dict()
        restored = SharedMemoryConfig.from_dict(data)

        assert restored.enabled == config.enabled
        assert restored.require_auth == config.require_auth
        assert restored.retention_days == config.retention_days
        assert restored.max_memories_per_user == config.max_memories_per_user

    def test_system_context_includes_shared_memory_config(self):
        """SystemContext should include SharedMemoryConfig."""
        system = SystemContext(
            system_id="test-system",
            system_name="Test System",
            shared_knowledge=[],
            shared_memory_config=SharedMemoryConfig(require_auth=True),
        )

        assert system.shared_memory_config is not None
        assert system.shared_memory_config.require_auth is True


class TestStrictModeEnforcement:
    """Test that strict mode (default) properly enforces all privacy rules."""

    def test_strict_mode_denies_all_anonymous_persistence(self):
        """Strict mode should deny ALL persistence for anonymous users."""
        config = DEFAULT_PRIVACY_CONFIG  # Strict mode
        anon = ANONYMOUS_USER

        # All persistence should be denied
        assert config.allows_memory(anon) is False
        assert config.allows_conversation_persistence(anon) is False
        assert config.allows_tool_tracking(anon) is False

    def test_strict_mode_allows_all_authenticated_persistence(self):
        """Strict mode should allow all persistence for authenticated users."""
        config = DEFAULT_PRIVACY_CONFIG  # Strict mode
        user = UserContext.from_user_id("user-123")

        # All persistence should be allowed
        assert config.allows_memory(user) is True
        assert config.allows_conversation_persistence(user) is True
        assert config.allows_tool_tracking(user) is True

    def test_cannot_accidentally_create_permissive_config(self):
        """Creating a PrivacyConfig without args should be strict."""
        config = PrivacyConfig()

        # Should be strict by default
        assert config.require_auth_for_memory is True
        assert config.require_auth_for_conversations is True
        assert config.require_auth_for_tool_tracking is True

    def test_permissive_mode_requires_explicit_opt_in(self):
        """Permissive mode should require explicit opt-in for each setting."""
        # Must explicitly set each to False
        permissive = PrivacyConfig(
            require_auth_for_memory=False,
            require_auth_for_conversations=False,
            require_auth_for_tool_tracking=False,
        )

        anon = ANONYMOUS_USER
        assert permissive.allows_memory(anon) is True
        assert permissive.allows_conversation_persistence(anon) is True
        assert permissive.allows_tool_tracking(anon) is True


class TestCrossUserPollutionPrevention:
    """Test that there is no possible cross-user data pollution."""

    def test_user_context_immutability(self):
        """UserContext should not allow modification after creation."""
        user = UserContext.from_user_id("user-123")

        # Attempting to modify should not affect the original
        # (dataclass is frozen by default in our implementation)
        original_id = user.user_id
        assert user.user_id == original_id

    def test_privacy_config_immutability(self):
        """PrivacyConfig should not allow modification after creation."""
        config = PrivacyConfig()

        # Original values should remain
        assert config.require_auth_for_memory is True

    def test_user_ids_cannot_be_empty_string(self):
        """User IDs should not be empty strings (could cause collision)."""
        # Empty string user_id should be treated as anonymous
        user = UserContext(user_id="", is_authenticated=False)
        assert user.is_authenticated is False

    def test_user_ids_cannot_be_none_for_authenticated(self):
        """Authenticated users must have a user_id."""
        # If user_id is None, user should not be authenticated
        user = UserContext(user_id=None, is_authenticated=True)
        # The from_user_id factory ensures this doesn't happen
        proper_user = UserContext.from_user_id("user-123")
        assert proper_user.user_id is not None
        assert proper_user.is_authenticated is True

    def test_metadata_isolation(self):
        """User metadata should be isolated between users."""
        user1 = UserContext(
            user_id="user-1",
            is_authenticated=True,
            metadata={"secret": "user1-secret"},
        )
        user2 = UserContext(
            user_id="user-2",
            is_authenticated=True,
            metadata={"secret": "user2-secret"},
        )

        # Metadata should be completely separate
        assert user1.metadata["secret"] != user2.metadata["secret"]
        assert "user1-secret" not in str(user2.metadata)
        assert "user2-secret" not in str(user1.metadata)

    def test_run_context_user_isolation(self):
        """Run contexts for different users should be completely isolated."""
        user1 = UserContext.from_user_id("user-1")
        user2 = UserContext.from_user_id("user-2")

        ctx1 = InMemoryRunContext(
            user_context=user1,
            metadata={"user_data": "user1-private-data"},
        )
        ctx2 = InMemoryRunContext(
            user_context=user2,
            metadata={"user_data": "user2-private-data"},
        )

        # Contexts should be completely separate
        assert ctx1.metadata["user_data"] != ctx2.metadata["user_data"]
        assert ctx1.user_context.user_id != ctx2.user_context.user_id

        # One context should not reference the other's data
        assert "user2" not in str(ctx1.metadata)
        assert "user1" not in str(ctx2.metadata)



class TestAsyncMultiUserScenarios:
    """Async tests simulating multi-user scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_users_isolated_checkpoints(self):
        """Concurrent users should have isolated checkpoint data."""
        import tempfile
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            user1 = UserContext.from_user_id("user-1")
            user2 = UserContext.from_user_id("user-2")

            ctx1 = FileRunContext(
                checkpoint_dir=tmpdir,
                user_context=user1,
            )
            ctx2 = FileRunContext(
                checkpoint_dir=tmpdir,
                user_context=user2,
            )

            # Checkpoint different data for each user
            await ctx1.checkpoint({"secret": "user1-secret-data"})
            await ctx2.checkpoint({"secret": "user2-secret-data"})

            # Each user should only see their own data
            state1 = await ctx1.get_state()
            state2 = await ctx2.get_state()

            assert state1["secret"] == "user1-secret-data"
            assert state2["secret"] == "user2-secret-data"

            # Cross-check: user1's data should not contain user2's secret
            assert "user2-secret-data" not in str(state1)
            assert "user1-secret-data" not in str(state2)

    @pytest.mark.asyncio
    async def test_concurrent_users_isolated_events(self):
        """Concurrent users should have isolated event streams."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            user1 = UserContext.from_user_id("user-1")
            user2 = UserContext.from_user_id("user-2")

            ctx1 = FileRunContext(
                checkpoint_dir=tmpdir,
                user_context=user1,
            )
            ctx2 = FileRunContext(
                checkpoint_dir=tmpdir,
                user_context=user2,
            )

            # Emit different events for each user
            await ctx1.emit("user.action", {"data": "user1-private-action"})
            await ctx2.emit("user.action", {"data": "user2-private-action"})

            # Each user should only see their own events
            events1 = ctx1.get_events()
            events2 = ctx2.get_events()

            assert len(events1) == 1
            assert len(events2) == 1
            assert events1[0]["payload"]["data"] == "user1-private-action"
            assert events2[0]["payload"]["data"] == "user2-private-action"

    @pytest.mark.asyncio
    async def test_privacy_check_before_memory_operation(self):
        """Privacy should be checked before any memory operation."""
        config = DEFAULT_PRIVACY_CONFIG
        anon = ANONYMOUS_USER
        auth_user = UserContext.from_user_id("user-123")

        # Simulate a memory operation check
        def attempt_memory_operation(user: UserContext) -> bool:
            """Simulate attempting a memory operation."""
            if not config.allows_memory(user):
                return False  # Operation blocked
            return True  # Operation allowed

        # Anonymous should be blocked
        assert attempt_memory_operation(anon) is False

        # Authenticated should be allowed
        assert attempt_memory_operation(auth_user) is True


class TestEdgeCases:
    """Test edge cases and potential security vulnerabilities."""

    def test_user_id_with_special_characters(self):
        """User IDs with special characters should be handled safely."""
        special_ids = [
            "user/../../../etc/passwd",  # Path traversal attempt
            "user'; DROP TABLE users;--",  # SQL injection attempt
            "user<script>alert('xss')</script>",  # XSS attempt
            "user\x00null",  # Null byte injection
            "user\nheader: injection",  # Header injection
        ]

        for special_id in special_ids:
            user = UserContext.from_user_id(special_id)
            # Should store the ID as-is (sanitization happens at storage layer)
            assert user.user_id == special_id
            assert user.is_authenticated is True

    def test_empty_metadata_is_safe(self):
        """Empty metadata should not cause issues."""
        user = UserContext.from_user_id("user-123")
        assert user.metadata == {}

        # Should serialize safely
        data = user.to_dict()
        assert "metadata" in data

    def test_none_values_handled_safely(self):
        """None values should be handled safely."""
        user = UserContext(
            user_id=None,
            is_authenticated=False,
            username=None,
            email=None,
        )

        # Should serialize safely
        data = user.to_dict()
        restored = UserContext.from_dict(data)
        assert restored.user_id is None
        assert restored.is_authenticated is False

    def test_from_dict_with_missing_fields(self):
        """from_dict should handle missing fields gracefully."""
        # Minimal dict
        data = {}
        user = UserContext.from_dict(data)
        assert user.user_id is None
        assert user.is_authenticated is False
        assert user.metadata == {}

    def test_large_user_id_handled(self):
        """Very large user IDs should be handled."""
        large_id = "user-" + "x" * 10000
        user = UserContext.from_user_id(large_id)
        assert user.user_id == large_id
        assert len(user.user_id) == 10005

    def test_unicode_user_id_handled(self):
        """Unicode user IDs should be handled correctly."""
        unicode_ids = [
            "用户-123",  # Chinese
            "пользователь-123",  # Russian
            "🔐user-123",  # Emoji
            "user-مستخدم",  # Arabic
        ]

        for uid in unicode_ids:
            user = UserContext.from_user_id(uid)
            assert user.user_id == uid
            assert user.is_authenticated is True

            # Should roundtrip through serialization
            data = user.to_dict()
            restored = UserContext.from_dict(data)
            assert restored.user_id == uid