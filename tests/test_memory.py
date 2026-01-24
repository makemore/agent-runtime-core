"""
Tests for the cross-conversation memory system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent_runtime_core.memory import (
    MemoryManager,
    MemoryConfig,
    ExtractedMemory,
    RecalledMemory,
    MemoryEnabledAgent,
)
from agent_runtime_core.memory.mixin import with_memory
from agent_runtime_core.persistence.base import Fact, FactType, Scope
from agent_runtime_core.interfaces import RunResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_knowledge_store():
    """Create a mock knowledge store."""
    store = AsyncMock()
    store.list_facts = AsyncMock(return_value=[])
    store.get_fact_by_key = AsyncMock(return_value=None)
    store.save_fact = AsyncMock()
    store.delete_fact = AsyncMock(return_value=True)
    return store


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    # Default response for extraction
    client.generate = AsyncMock(return_value=MagicMock(
        message={"content": "[]"}
    ))
    return client


@pytest.fixture
def memory_manager(mock_knowledge_store, mock_llm_client):
    """Create a memory manager with mocks."""
    return MemoryManager(
        knowledge_store=mock_knowledge_store,
        llm_client=mock_llm_client,
        user_id="test-user",
    )


# =============================================================================
# MemoryConfig Tests
# =============================================================================


class TestMemoryConfig:
    """Tests for MemoryConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()
        
        assert config.extract_after_messages == 4
        assert config.extract_on_conversation_end is True
        assert config.max_facts_per_extraction == 5
        assert config.max_memories_to_recall == 10
        assert config.relevance_threshold == 0.5
        assert config.scope == Scope.GLOBAL
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            extract_after_messages=2,
            max_memories_to_recall=5,
            relevance_threshold=0.7,
            scope=Scope.PROJECT,
        )
        
        assert config.extract_after_messages == 2
        assert config.max_memories_to_recall == 5
        assert config.relevance_threshold == 0.7
        assert config.scope == Scope.PROJECT


# =============================================================================
# MemoryManager Tests
# =============================================================================


class TestMemoryManager:
    """Tests for MemoryManager."""
    
    @pytest.mark.asyncio
    async def test_extract_memories_empty_messages(self, memory_manager):
        """Test extraction with empty messages returns empty list."""
        result = await memory_manager.extract_memories([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_extract_memories_parses_llm_response(
        self, mock_knowledge_store, mock_llm_client
    ):
        """Test that extraction parses LLM response correctly."""
        # Setup LLM to return facts
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(
            message={"content": '''[
                {"key": "user_name", "value": "Alice", "type": "user", "confidence": 1.0},
                {"key": "preferred_theme", "value": "dark", "type": "preference", "confidence": 0.9}
            ]'''}
        ))
        
        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
        )
        
        messages = [
            {"role": "user", "content": "My name is Alice and I prefer dark mode."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]
        
        result = await manager.extract_memories(messages)
        
        assert len(result) == 2
        assert result[0].key == "user_name"
        assert result[0].value == "Alice"
        assert result[0].fact_type == FactType.USER
        assert result[1].key == "preferred_theme"
        assert result[1].value == "dark"
        assert result[1].fact_type == FactType.PREFERENCE
    
    @pytest.mark.asyncio
    async def test_extract_memories_handles_code_blocks(
        self, mock_knowledge_store, mock_llm_client
    ):
        """Test that extraction handles markdown code blocks."""
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(
            message={"content": '''```json
[{"key": "user_name", "value": "Bob", "type": "user", "confidence": 1.0}]
```'''}
        ))
        
        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
        )
        
        result = await manager.extract_memories([
            {"role": "user", "content": "I'm Bob"}
        ])
        
        assert len(result) == 1
        assert result[0].key == "user_name"
        assert result[0].value == "Bob"
    
    @pytest.mark.asyncio
    async def test_extract_memories_saves_to_store(
        self, mock_knowledge_store, mock_llm_client
    ):
        """Test that extracted memories are saved to store."""
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(
            message={"content": '[{"key": "user_name", "value": "Alice", "type": "user", "confidence": 1.0}]'}
        ))
        
        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="user-123",
        )
        
        await manager.extract_memories([
            {"role": "user", "content": "I'm Alice"}
        ])
        
        # Verify save_fact was called
        mock_knowledge_store.save_fact.assert_called_once()
        saved_fact = mock_knowledge_store.save_fact.call_args[0][0]
        assert saved_fact.key == "user:user-123:user_name"
        assert saved_fact.value == "Alice"

    @pytest.mark.asyncio
    async def test_extract_memories_no_save(
        self, mock_knowledge_store, mock_llm_client
    ):
        """Test extraction without saving."""
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(
            message={"content": '[{"key": "test", "value": "value", "type": "user", "confidence": 1.0}]'}
        ))

        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
        )

        result = await manager.extract_memories(
            [{"role": "user", "content": "test"}],
            save=False,
        )

        assert len(result) == 1
        mock_knowledge_store.save_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_memories_empty_store(self, memory_manager):
        """Test recall with empty store returns empty list."""
        result = await memory_manager.recall_memories()
        assert result == []

    @pytest.mark.asyncio
    async def test_recall_memories_returns_all_without_query(
        self, mock_knowledge_store, mock_llm_client
    ):
        """Test recall without query returns all memories."""
        # Setup store with facts
        mock_knowledge_store.list_facts = AsyncMock(return_value=[
            Fact(id=uuid4(), key="user:test-user:name", value="Alice", fact_type=FactType.USER),
            Fact(id=uuid4(), key="user:test-user:theme", value="dark", fact_type=FactType.PREFERENCE),
        ])

        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
        )

        result = await manager.recall_memories()

        assert len(result) == 2
        assert result[0].key == "name"
        assert result[0].value == "Alice"
        assert result[1].key == "theme"
        assert result[1].value == "dark"

    @pytest.mark.asyncio
    async def test_recall_memories_with_query_ranks_relevance(
        self, mock_knowledge_store, mock_llm_client
    ):
        """Test recall with query uses LLM to rank relevance."""
        mock_knowledge_store.list_facts = AsyncMock(return_value=[
            Fact(id=uuid4(), key="user:test-user:name", value="Alice", fact_type=FactType.USER),
            Fact(id=uuid4(), key="user:test-user:theme", value="dark", fact_type=FactType.PREFERENCE),
        ])

        # LLM returns relevance scores
        mock_llm_client.generate = AsyncMock(return_value=MagicMock(
            message={"content": '[{"key": "theme", "relevance": 0.9}, {"key": "name", "relevance": 0.3}]'}
        ))

        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
            config=MemoryConfig(relevance_threshold=0.5),
        )

        result = await manager.recall_memories(query="What theme does the user prefer?")

        # Only theme should be returned (name is below threshold)
        assert len(result) == 1
        assert result[0].key == "theme"
        assert result[0].relevance == 0.9

    @pytest.mark.asyncio
    async def test_get_memory(self, mock_knowledge_store, mock_llm_client):
        """Test getting a specific memory."""
        mock_knowledge_store.get_fact_by_key = AsyncMock(return_value=Fact(
            id=uuid4(),
            key="user:test-user:name",
            value="Alice",
            fact_type=FactType.USER,
        ))

        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
        )

        result = await manager.get_memory("name")

        assert result == "Alice"
        mock_knowledge_store.get_fact_by_key.assert_called_with(
            "user:test-user:name", Scope.GLOBAL
        )

    @pytest.mark.asyncio
    async def test_set_memory(self, mock_knowledge_store, mock_llm_client):
        """Test manually setting a memory."""
        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
        )

        await manager.set_memory("favorite_color", "blue", FactType.PREFERENCE)

        mock_knowledge_store.save_fact.assert_called_once()
        saved_fact = mock_knowledge_store.save_fact.call_args[0][0]
        assert saved_fact.key == "user:test-user:favorite_color"
        assert saved_fact.value == "blue"
        assert saved_fact.fact_type == FactType.PREFERENCE

    @pytest.mark.asyncio
    async def test_delete_memory(self, mock_knowledge_store, mock_llm_client):
        """Test deleting a memory."""
        fact_id = uuid4()
        mock_knowledge_store.get_fact_by_key = AsyncMock(return_value=Fact(
            id=fact_id,
            key="user:test-user:name",
            value="Alice",
            fact_type=FactType.USER,
        ))

        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
        )

        result = await manager.delete_memory("name")

        assert result is True
        mock_knowledge_store.delete_fact.assert_called_with(fact_id, Scope.GLOBAL)

    @pytest.mark.asyncio
    async def test_clear_memories(self, mock_knowledge_store, mock_llm_client):
        """Test clearing all memories for a user."""
        fact1_id = uuid4()
        fact2_id = uuid4()
        mock_knowledge_store.list_facts = AsyncMock(return_value=[
            Fact(id=fact1_id, key="user:test-user:name", value="Alice", fact_type=FactType.USER),
            Fact(id=fact2_id, key="user:test-user:theme", value="dark", fact_type=FactType.PREFERENCE),
            Fact(id=uuid4(), key="user:other-user:name", value="Bob", fact_type=FactType.USER),
        ])

        manager = MemoryManager(
            knowledge_store=mock_knowledge_store,
            llm_client=mock_llm_client,
            user_id="test-user",
        )

        count = await manager.clear_memories()

        assert count == 2
        assert mock_knowledge_store.delete_fact.call_count == 2


class TestMemoryManagerFormatting:
    """Tests for memory formatting."""

    def test_format_memories_list_style(self, memory_manager):
        """Test list-style formatting."""
        memories = [
            RecalledMemory(key="name", value="Alice", fact_type=FactType.USER),
            RecalledMemory(key="theme", value="dark", fact_type=FactType.PREFERENCE),
        ]

        result = memory_manager.format_memories_for_prompt(memories, "list")

        assert "Remembered information about the user:" in result
        assert "- name: Alice" in result
        assert "- theme: dark" in result

    def test_format_memories_prose_style(self, memory_manager):
        """Test prose-style formatting."""
        memories = [
            RecalledMemory(key="name", value="Alice", fact_type=FactType.USER),
        ]

        result = memory_manager.format_memories_for_prompt(memories, "prose")

        assert "What I remember:" in result
        assert "name is Alice" in result

    def test_format_memories_structured_style(self, memory_manager):
        """Test structured formatting."""
        memories = [
            RecalledMemory(key="name", value="Alice", fact_type=FactType.USER),
        ]

        result = memory_manager.format_memories_for_prompt(memories, "structured")

        assert "User context:" in result
        assert '"name": "Alice"' in result

    def test_format_memories_empty(self, memory_manager):
        """Test formatting empty memories."""
        result = memory_manager.format_memories_for_prompt([], "list")
        assert result == ""


# =============================================================================
# MemoryEnabledAgent Mixin Tests
# =============================================================================


class TestMemoryEnabledAgent:
    """Tests for the MemoryEnabledAgent mixin."""

    def test_mixin_default_disabled(self):
        """Test that memory is disabled by default."""
        class TestAgent(MemoryEnabledAgent):
            pass

        agent = TestAgent()
        assert agent.memory_enabled is False

    def test_mixin_can_be_enabled(self):
        """Test that memory can be enabled."""
        class TestAgent(MemoryEnabledAgent):
            memory_enabled = True

        agent = TestAgent()
        assert agent.memory_enabled is True

    def test_configure_memory(self, mock_knowledge_store, mock_llm_client):
        """Test configuring memory on an agent."""
        class TestAgent(MemoryEnabledAgent):
            memory_enabled = True

            def get_llm_client(self):
                return mock_llm_client

        agent = TestAgent()
        agent.configure_memory(mock_knowledge_store)

        assert agent._memory_manager is not None

    @pytest.mark.asyncio
    async def test_recall_memories_for_context(self):
        """Test recalling memories for a context."""
        # Create fresh mocks for this test
        store = AsyncMock()
        store.list_facts = AsyncMock(return_value=[
            Fact(id=uuid4(), key="user:user-123:name", value="Alice", fact_type=FactType.USER),
        ])
        store.get_fact_by_key = AsyncMock(return_value=None)

        llm = AsyncMock()
        # Mock LLM to return relevance scores for the memories
        llm.generate = AsyncMock(return_value=MagicMock(
            message={"content": '[{"key": "name", "relevance": 0.9}]'}
        ))

        class TestAgent(MemoryEnabledAgent):
            memory_enabled = True

            def get_llm_client(self):
                return llm

        agent = TestAgent()
        agent.configure_memory(store, llm_client=llm)
        # Set user_id on the memory manager
        agent._memory_manager._user_id = "user-123"

        # Create a mock context
        ctx = MagicMock()
        ctx.metadata = {"user_id": "user-123"}
        ctx.input_messages = [{"role": "user", "content": "Hello"}]

        memories = await agent.recall_memories_for_context(ctx)

        assert len(memories) == 1
        assert memories[0].key == "name"
        assert memories[0].relevance == 0.9


class TestWithMemoryFunction:
    """Tests for the with_memory convenience function."""

    def test_creates_memory_enabled_class(self, mock_knowledge_store, mock_llm_client):
        """Test that with_memory creates a memory-enabled class."""
        from agent_runtime_core import ToolCallingAgent, ToolRegistry

        class SimpleAgent(ToolCallingAgent):
            @property
            def key(self):
                return "simple"

            @property
            def system_prompt(self):
                return "You are helpful."

            @property
            def tools(self):
                return ToolRegistry()

        MemoryAgent = with_memory(SimpleAgent, mock_knowledge_store, llm_client=mock_llm_client)

        assert MemoryAgent.__name__ == "MemoryEnabledSimpleAgent"

        agent = MemoryAgent()
        assert agent.memory_enabled is True
        assert agent._memory_manager is not None

