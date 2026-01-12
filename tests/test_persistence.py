"""Tests for the persistence module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from agent_runtime_core.persistence import (
    # Abstract interfaces
    MemoryStore,
    ConversationStore,
    TaskStore,
    PreferencesStore,
    Scope,
    # Data classes
    Conversation,
    ConversationMessage,
    ToolCall,
    TaskList,
    Task,
    TaskState,
    # File implementations
    FileMemoryStore,
    FileConversationStore,
    FileTaskStore,
    FilePreferencesStore,
    # Manager
    PersistenceManager,
    PersistenceConfig,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def memory_store(temp_dir):
    """Create a FileMemoryStore with temp directory."""
    return FileMemoryStore(project_dir=temp_dir)


@pytest.fixture
def conversation_store(temp_dir):
    """Create a FileConversationStore with temp directory."""
    return FileConversationStore(project_dir=temp_dir)


@pytest.fixture
def task_store(temp_dir):
    """Create a FileTaskStore with temp directory."""
    return FileTaskStore(project_dir=temp_dir)


@pytest.fixture
def preferences_store(temp_dir):
    """Create a FilePreferencesStore with temp directory."""
    return FilePreferencesStore(project_dir=temp_dir)


class TestFileMemoryStore:
    """Tests for FileMemoryStore."""
    
    async def test_set_and_get(self, memory_store):
        """Test setting and getting a value."""
        await memory_store.set("test_key", "test_value")
        result = await memory_store.get("test_key")
        assert result == "test_value"
    
    async def test_get_nonexistent(self, memory_store):
        """Test getting a nonexistent key."""
        result = await memory_store.get("nonexistent")
        assert result is None
    
    async def test_delete(self, memory_store):
        """Test deleting a key."""
        await memory_store.set("to_delete", "value")
        assert await memory_store.delete("to_delete") is True
        assert await memory_store.get("to_delete") is None
        assert await memory_store.delete("to_delete") is False
    
    async def test_list_keys(self, memory_store):
        """Test listing keys."""
        await memory_store.set("key1", "value1")
        await memory_store.set("key2", "value2")
        await memory_store.set("other", "value3")
        
        all_keys = await memory_store.list_keys()
        assert set(all_keys) == {"key1", "key2", "other"}
        
        filtered = await memory_store.list_keys(prefix="key")
        assert set(filtered) == {"key1", "key2"}
    
    async def test_clear(self, memory_store):
        """Test clearing all keys."""
        await memory_store.set("key1", "value1")
        await memory_store.set("key2", "value2")
        await memory_store.clear()
        
        keys = await memory_store.list_keys()
        assert keys == []
    
    async def test_complex_values(self, memory_store):
        """Test storing complex values."""
        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": "b"},
            "number": 42,
        }
        await memory_store.set("complex", complex_value)
        result = await memory_store.get("complex")
        assert result == complex_value


class TestFileConversationStore:
    """Tests for FileConversationStore."""
    
    async def test_save_and_get(self, conversation_store):
        """Test saving and getting a conversation."""
        conv_id = uuid4()
        conversation = Conversation(
            id=conv_id,
            title="Test Conversation",
            messages=[
                ConversationMessage(
                    id=uuid4(),
                    role="user",
                    content="Hello!",
                ),
                ConversationMessage(
                    id=uuid4(),
                    role="assistant",
                    content="Hi there!",
                ),
            ],
            agent_key="test-agent",
        )
        
        await conversation_store.save(conversation)
        result = await conversation_store.get(conv_id)
        
        assert result is not None
        assert result.id == conv_id
        assert result.title == "Test Conversation"
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].content == "Hi there!"
    
    async def test_conversation_with_tool_calls(self, conversation_store):
        """Test conversation with tool calls."""
        conv_id = uuid4()
        conversation = Conversation(
            id=conv_id,
            messages=[
                ConversationMessage(
                    id=uuid4(),
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            name="get_weather",
                            arguments={"location": "NYC"},
                        ),
                    ],
                ),
            ],
        )
        
        await conversation_store.save(conversation)
        result = await conversation_store.get(conv_id)
        
        assert len(result.messages[0].tool_calls) == 1
        assert result.messages[0].tool_calls[0].name == "get_weather"

    async def test_add_message(self, conversation_store):
        """Test adding a message to existing conversation."""
        conv_id = uuid4()
        conversation = Conversation(id=conv_id, title="Test")
        await conversation_store.save(conversation)

        message = ConversationMessage(
            id=uuid4(),
            role="user",
            content="New message",
        )
        await conversation_store.add_message(conv_id, message)

        result = await conversation_store.get(conv_id)
        assert len(result.messages) == 1
        assert result.messages[0].content == "New message"

    async def test_list_conversations(self, conversation_store):
        """Test listing conversations."""
        for i in range(3):
            conv = Conversation(id=uuid4(), title=f"Conv {i}", agent_key="agent-a")
            await conversation_store.save(conv)

        conv_b = Conversation(id=uuid4(), title="Conv B", agent_key="agent-b")
        await conversation_store.save(conv_b)

        all_convs = await conversation_store.list_conversations()
        assert len(all_convs) == 4

        agent_a_convs = await conversation_store.list_conversations(agent_key="agent-a")
        assert len(agent_a_convs) == 3

    async def test_delete_conversation(self, conversation_store):
        """Test deleting a conversation."""
        conv_id = uuid4()
        conversation = Conversation(id=conv_id, title="To Delete")
        await conversation_store.save(conversation)

        assert await conversation_store.delete(conv_id) is True
        assert await conversation_store.get(conv_id) is None
        assert await conversation_store.delete(conv_id) is False


class TestFileTaskStore:
    """Tests for FileTaskStore."""

    async def test_save_and_get(self, task_store):
        """Test saving and getting a task list."""
        task_list_id = uuid4()
        task_list = TaskList(
            id=task_list_id,
            name="My Tasks",
            tasks=[
                Task(id=uuid4(), name="Task 1", state=TaskState.NOT_STARTED),
                Task(id=uuid4(), name="Task 2", state=TaskState.IN_PROGRESS),
            ],
        )

        await task_store.save(task_list)
        result = await task_store.get(task_list_id)

        assert result is not None
        assert result.name == "My Tasks"
        assert len(result.tasks) == 2
        assert result.tasks[0].state == TaskState.NOT_STARTED
        assert result.tasks[1].state == TaskState.IN_PROGRESS

    async def test_update_task(self, task_store):
        """Test updating a task."""
        task_id = uuid4()
        task_list_id = uuid4()
        task_list = TaskList(
            id=task_list_id,
            name="Tasks",
            tasks=[Task(id=task_id, name="Original", state=TaskState.NOT_STARTED)],
        )
        await task_store.save(task_list)

        await task_store.update_task(
            task_list_id,
            task_id,
            state=TaskState.COMPLETE,
            name="Updated",
        )

        result = await task_store.get(task_list_id)
        assert result.tasks[0].name == "Updated"
        assert result.tasks[0].state == TaskState.COMPLETE

    async def test_get_by_conversation(self, task_store):
        """Test getting task list by conversation ID."""
        conv_id = uuid4()
        task_list = TaskList(
            id=uuid4(),
            name="Conv Tasks",
            conversation_id=conv_id,
        )
        await task_store.save(task_list)

        result = await task_store.get_by_conversation(conv_id)
        assert result is not None
        assert result.name == "Conv Tasks"


class TestFilePreferencesStore:
    """Tests for FilePreferencesStore."""

    async def test_set_and_get(self, preferences_store):
        """Test setting and getting preferences."""
        # Use PROJECT scope to use temp_dir
        await preferences_store.set("theme", "dark", scope=Scope.PROJECT)
        result = await preferences_store.get("theme", scope=Scope.PROJECT)
        assert result == "dark"

    async def test_get_all(self, preferences_store):
        """Test getting all preferences."""
        # Use PROJECT scope to use temp_dir
        await preferences_store.set("pref1", "value1", scope=Scope.PROJECT)
        await preferences_store.set("pref2", "value2", scope=Scope.PROJECT)

        all_prefs = await preferences_store.get_all(scope=Scope.PROJECT)
        assert all_prefs == {"pref1": "value1", "pref2": "value2"}

    async def test_delete(self, preferences_store):
        """Test deleting a preference."""
        # Use PROJECT scope to use temp_dir
        await preferences_store.set("to_delete", "value", scope=Scope.PROJECT)
        assert await preferences_store.delete("to_delete", scope=Scope.PROJECT) is True
        assert await preferences_store.get("to_delete", scope=Scope.PROJECT) is None


class TestPersistenceManager:
    """Tests for PersistenceManager."""

    def test_default_config(self, temp_dir):
        """Test manager with default config."""
        config = PersistenceConfig(project_dir=temp_dir)
        manager = PersistenceManager(config)

        assert isinstance(manager.memory, FileMemoryStore)
        assert isinstance(manager.conversations, FileConversationStore)
        assert isinstance(manager.tasks, FileTaskStore)
        assert isinstance(manager.preferences, FilePreferencesStore)

    async def test_manager_operations(self, temp_dir):
        """Test manager operations."""
        config = PersistenceConfig(project_dir=temp_dir)
        manager = PersistenceManager(config)

        # Test memory
        await manager.memory.set("key", "value")
        assert await manager.memory.get("key") == "value"

        # Test conversations
        conv = Conversation(id=uuid4(), title="Test")
        await manager.conversations.save(conv)
        assert await manager.conversations.get(conv.id) is not None

        await manager.close()

    def test_pre_instantiated_stores(self, temp_dir):
        """Test passing pre-instantiated store instances."""
        memory_store = FileMemoryStore(project_dir=temp_dir)
        conversation_store = FileConversationStore(project_dir=temp_dir)

        config = PersistenceConfig(
            memory_store=memory_store,
            conversation_store=conversation_store,
        )
        manager = PersistenceManager(config)

        # Should return the exact same instances
        assert manager.memory is memory_store
        assert manager.conversations is conversation_store

    def test_factory_functions(self, temp_dir):
        """Test using factory functions."""
        call_count = {"memory": 0, "conversation": 0}

        def memory_factory():
            call_count["memory"] += 1
            return FileMemoryStore(project_dir=temp_dir)

        def conversation_factory():
            call_count["conversation"] += 1
            return FileConversationStore(project_dir=temp_dir)

        config = PersistenceConfig(
            memory_store_factory=memory_factory,
            conversation_store_factory=conversation_factory,
        )
        manager = PersistenceManager(config)

        # Factory should not be called until property is accessed
        assert call_count["memory"] == 0

        # Access memory - factory should be called once
        _ = manager.memory
        assert call_count["memory"] == 1

        # Access again - should return cached instance, not call factory again
        _ = manager.memory
        assert call_count["memory"] == 1

    def test_instance_takes_precedence_over_factory(self, temp_dir):
        """Test that pre-instantiated stores take precedence over factories."""
        memory_store = FileMemoryStore(project_dir=temp_dir)
        factory_called = {"called": False}

        def memory_factory():
            factory_called["called"] = True
            return FileMemoryStore(project_dir=temp_dir)

        config = PersistenceConfig(
            memory_store=memory_store,
            memory_store_factory=memory_factory,
        )
        manager = PersistenceManager(config)

        # Should use instance, not factory
        assert manager.memory is memory_store
        assert factory_called["called"] is False

    def test_optional_stores_none_by_default(self, temp_dir):
        """Test that optional stores return None when not configured."""
        config = PersistenceConfig(project_dir=temp_dir)
        manager = PersistenceManager(config)

        # Optional stores should be None
        assert manager.knowledge is None
        assert manager.audit is None
        assert manager.has_knowledge() is False
        assert manager.has_audit() is False

        # Core stores should still work
        assert manager.memory is not None
        assert manager.conversations is not None


class TestNewModels:
    """Tests for enhanced models."""

    def test_conversation_message_branching(self):
        """Test ConversationMessage with branching fields."""
        parent_id = uuid4()
        branch_id = uuid4()

        msg = ConversationMessage(
            id=uuid4(),
            role="assistant",
            content="Hello!",
            parent_message_id=parent_id,
            branch_id=branch_id,
        )

        assert msg.parent_message_id == parent_id
        assert msg.branch_id == branch_id

    def test_conversation_forking(self):
        """Test Conversation with forking fields."""
        parent_conv_id = uuid4()
        branch_id = uuid4()

        conv = Conversation(
            id=uuid4(),
            title="Forked conversation",
            parent_conversation_id=parent_conv_id,
            active_branch_id=branch_id,
        )

        assert conv.parent_conversation_id == parent_conv_id
        assert conv.active_branch_id == branch_id

    def test_task_enhanced_fields(self):
        """Test Task with enhanced fields."""
        dep1 = uuid4()
        dep2 = uuid4()

        task = Task(
            id=uuid4(),
            name="Complex task",
            description="A task with dependencies",
            dependencies=[dep1, dep2],
            priority=10,
            due_at=datetime(2024, 12, 31),
            checkpoint_data={"step": 5, "progress": 0.5},
            attempts=2,
            last_error="Temporary failure",
        )

        assert task.dependencies == [dep1, dep2]
        assert task.priority == 10
        assert task.due_at == datetime(2024, 12, 31)
        assert task.checkpoint_data == {"step": 5, "progress": 0.5}
        assert task.attempts == 2
        assert task.last_error == "Temporary failure"


class TestKnowledgeModels:
    """Tests for knowledge base models."""

    def test_fact_model(self):
        """Test Fact model."""
        from agent_runtime_core.persistence import Fact, FactType

        fact = Fact(
            id=uuid4(),
            key="user.name",
            value="Alice",
            fact_type=FactType.USER,
            confidence=0.95,
            source="conversation_123",
        )

        assert fact.key == "user.name"
        assert fact.value == "Alice"
        assert fact.fact_type == FactType.USER
        assert fact.confidence == 0.95
        assert fact.source == "conversation_123"

    def test_summary_model(self):
        """Test Summary model."""
        from agent_runtime_core.persistence import Summary

        conv_id = uuid4()
        summary = Summary(
            id=uuid4(),
            content="User discussed Python project setup.",
            conversation_id=conv_id,
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 11, 0),
        )

        assert summary.content == "User discussed Python project setup."
        assert summary.conversation_id == conv_id

    def test_embedding_model(self):
        """Test Embedding model."""
        from agent_runtime_core.persistence import Embedding

        embedding = Embedding(
            id=uuid4(),
            vector=[0.1, 0.2, 0.3, 0.4],
            content="Hello world",
            content_type="text",
            model="text-embedding-ada-002",
            dimensions=4,
        )

        assert embedding.vector == [0.1, 0.2, 0.3, 0.4]
        assert embedding.content == "Hello world"
        assert embedding.dimensions == 4


class TestAuditModels:
    """Tests for audit models."""

    def test_audit_entry_model(self):
        """Test AuditEntry model."""
        from agent_runtime_core.persistence import AuditEntry, AuditEventType

        entry = AuditEntry(
            id=uuid4(),
            event_type=AuditEventType.TOOL_CALL,
            action="Called search tool",
            details={"tool": "search", "query": "python docs"},
            actor_type="agent",
            request_id="req_123",
        )

        assert entry.event_type == AuditEventType.TOOL_CALL
        assert entry.action == "Called search tool"
        assert entry.details["tool"] == "search"

    def test_error_record_model(self):
        """Test ErrorRecord model."""
        from agent_runtime_core.persistence import ErrorRecord, ErrorSeverity

        error = ErrorRecord(
            id=uuid4(),
            severity=ErrorSeverity.ERROR,
            error_type="ValueError",
            message="Invalid input",
            stack_trace="Traceback...",
            context={"input": "bad data"},
        )

        assert error.severity == ErrorSeverity.ERROR
        assert error.error_type == "ValueError"
        assert error.message == "Invalid input"
        assert error.resolved is False

    def test_performance_metric_model(self):
        """Test PerformanceMetric model."""
        from agent_runtime_core.persistence import PerformanceMetric

        metric = PerformanceMetric(
            id=uuid4(),
            name="llm_latency",
            value=1250.5,
            unit="ms",
            tags={"model": "gpt-4", "provider": "openai"},
        )

        assert metric.name == "llm_latency"
        assert metric.value == 1250.5
        assert metric.unit == "ms"
        assert metric.tags["model"] == "gpt-4"

