"""
File-based implementations of persistence stores.

These implementations store data in hidden directories:
- Global: ~/.agent_runtime/
- Project: ./.agent_runtime/

Data is stored as JSON files for easy inspection and debugging.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict
from uuid import UUID

from agent_runtime_core.persistence.base import (
    MemoryStore,
    ConversationStore,
    TaskStore,
    PreferencesStore,
    KnowledgeStore,
    Scope,
    Conversation,
    ConversationMessage,
    ToolCall,
    ToolResult,
    TaskList,
    Task,
    TaskState,
    Fact,
    FactType,
    Summary,
    Embedding,
)


def _get_base_path(scope: Scope, project_dir: Optional[Path] = None) -> Path:
    """Get the base path for a given scope."""
    if scope == Scope.GLOBAL:
        return Path.home() / ".agent_runtime"
    elif scope == Scope.PROJECT:
        base = project_dir or Path.cwd()
        return base / ".agent_runtime"
    else:
        raise ValueError(f"Cannot get path for scope: {scope}")


def _ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


class _JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for our data types."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, TaskState):
            return obj.value
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        return super().default(obj)


def _json_dumps(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, cls=_JSONEncoder, indent=2)


def _parse_datetime(value: Any) -> datetime:
    """Parse a datetime from string or return as-is."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


def _parse_uuid(value: Any) -> UUID:
    """Parse a UUID from string or return as-is."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    return value


class FileMemoryStore(MemoryStore):
    """
    File-based memory store.

    Stores key-value pairs in JSON files:
    - {base_path}/memory/{key}.json
    """

    def __init__(self, project_dir: Optional[Path] = None):
        self._project_dir = project_dir

    def _get_memory_path(self, scope: Scope) -> Path:
        return _get_base_path(scope, self._project_dir) / "memory"

    def _get_key_path(self, key: str, scope: Scope) -> Path:
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._get_memory_path(scope) / f"{safe_key}.json"

    async def get(self, key: str, scope: Scope = Scope.PROJECT) -> Optional[Any]:
        path = self._get_key_path(key, scope)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("value")
        except (json.JSONDecodeError, IOError):
            return None

    async def set(self, key: str, value: Any, scope: Scope = Scope.PROJECT) -> None:
        path = self._get_key_path(key, scope)
        _ensure_dir(path.parent)
        with open(path, "w") as f:
            f.write(_json_dumps({
                "key": key,
                "value": value,
                "updated_at": datetime.utcnow(),
            }))

    async def delete(self, key: str, scope: Scope = Scope.PROJECT) -> bool:
        path = self._get_key_path(key, scope)
        if path.exists():
            path.unlink()
            return True
        return False

    async def list_keys(self, scope: Scope = Scope.PROJECT, prefix: Optional[str] = None) -> list[str]:
        memory_path = self._get_memory_path(scope)
        if not memory_path.exists():
            return []
        keys = []
        for file in memory_path.glob("*.json"):
            key = file.stem
            if prefix is None or key.startswith(prefix):
                keys.append(key)
        return sorted(keys)

    async def clear(self, scope: Scope = Scope.PROJECT) -> None:
        memory_path = self._get_memory_path(scope)
        if memory_path.exists():
            for file in memory_path.glob("*.json"):
                file.unlink()


class FileConversationStore(ConversationStore):
    """
    File-based conversation store.

    Stores conversations in JSON files:
    - {base_path}/conversations/{conversation_id}.json
    """

    def __init__(self, project_dir: Optional[Path] = None):
        self._project_dir = project_dir

    def _get_conversations_path(self, scope: Scope) -> Path:
        return _get_base_path(scope, self._project_dir) / "conversations"

    def _get_conversation_path(self, conversation_id: UUID, scope: Scope) -> Path:
        return self._get_conversations_path(scope) / f"{conversation_id}.json"

    def _serialize_conversation(self, conversation: Conversation) -> dict:
        """Serialize a conversation to a dict."""
        return {
            "id": str(conversation.id),
            "title": conversation.title,
            "messages": [self._serialize_message(m) for m in conversation.messages],
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "metadata": conversation.metadata,
            "agent_key": conversation.agent_key,
            "summary": conversation.summary,
        }

    def _serialize_message(self, message: ConversationMessage) -> dict:
        """Serialize a message to a dict."""
        return {
            "id": str(message.id),
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "timestamp": tc.timestamp.isoformat(),
                }
                for tc in message.tool_calls
            ],
            "tool_call_id": message.tool_call_id,
            "model": message.model,
            "usage": message.usage,
            "metadata": message.metadata,
        }

    def _deserialize_conversation(self, data: dict) -> Conversation:
        """Deserialize a conversation from a dict."""
        return Conversation(
            id=_parse_uuid(data["id"]),
            title=data.get("title"),
            messages=[self._deserialize_message(m) for m in data.get("messages", [])],
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            metadata=data.get("metadata", {}),
            agent_key=data.get("agent_key"),
            summary=data.get("summary"),
        )

    def _deserialize_message(self, data: dict) -> ConversationMessage:
        """Deserialize a message from a dict."""
        return ConversationMessage(
            id=_parse_uuid(data["id"]),
            role=data["role"],
            content=data["content"],
            timestamp=_parse_datetime(data["timestamp"]),
            tool_calls=[
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                    timestamp=_parse_datetime(tc["timestamp"]),
                )
                for tc in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            model=data.get("model"),
            usage=data.get("usage", {}),
            metadata=data.get("metadata", {}),
        )

    async def save(self, conversation: Conversation, scope: Scope = Scope.PROJECT) -> None:
        path = self._get_conversation_path(conversation.id, scope)
        _ensure_dir(path.parent)
        conversation.updated_at = datetime.utcnow()
        with open(path, "w") as f:
            f.write(_json_dumps(self._serialize_conversation(conversation)))

    async def get(self, conversation_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[Conversation]:
        path = self._get_conversation_path(conversation_id, scope)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return self._deserialize_conversation(data)
        except (json.JSONDecodeError, IOError):
            return None

    async def delete(self, conversation_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        path = self._get_conversation_path(conversation_id, scope)
        if path.exists():
            path.unlink()
            return True
        return False

    async def list_conversations(
        self,
        scope: Scope = Scope.PROJECT,
        limit: int = 100,
        offset: int = 0,
        agent_key: Optional[str] = None,
    ) -> list[Conversation]:
        conversations_path = self._get_conversations_path(scope)
        if not conversations_path.exists():
            return []

        conversations = []
        for file in conversations_path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    conv = self._deserialize_conversation(data)
                    if agent_key is None or conv.agent_key == agent_key:
                        conversations.append(conv)
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by updated_at descending
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations[offset:offset + limit]

    async def add_message(
        self,
        conversation_id: UUID,
        message: ConversationMessage,
        scope: Scope = Scope.PROJECT,
    ) -> None:
        conversation = await self.get(conversation_id, scope)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        conversation.messages.append(message)
        await self.save(conversation, scope)

    async def get_messages(
        self,
        conversation_id: UUID,
        scope: Scope = Scope.PROJECT,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> list[ConversationMessage]:
        conversation = await self.get(conversation_id, scope)
        if conversation is None:
            return []

        messages = conversation.messages
        if before:
            messages = [m for m in messages if m.timestamp < before]
        if limit:
            messages = messages[-limit:]
        return messages


class FileTaskStore(TaskStore):
    """
    File-based task store.

    Stores task lists in JSON files:
    - {base_path}/tasks/{task_list_id}.json
    """

    def __init__(self, project_dir: Optional[Path] = None):
        self._project_dir = project_dir

    def _get_tasks_path(self, scope: Scope) -> Path:
        return _get_base_path(scope, self._project_dir) / "tasks"

    def _get_task_list_path(self, task_list_id: UUID, scope: Scope) -> Path:
        return self._get_tasks_path(scope) / f"{task_list_id}.json"

    def _serialize_task_list(self, task_list: TaskList) -> dict:
        return {
            "id": str(task_list.id),
            "name": task_list.name,
            "tasks": [
                {
                    "id": str(t.id),
                    "name": t.name,
                    "description": t.description,
                    "state": t.state.value,
                    "parent_id": str(t.parent_id) if t.parent_id else None,
                    "created_at": t.created_at.isoformat(),
                    "updated_at": t.updated_at.isoformat(),
                    "metadata": t.metadata,
                }
                for t in task_list.tasks
            ],
            "created_at": task_list.created_at.isoformat(),
            "updated_at": task_list.updated_at.isoformat(),
            "conversation_id": str(task_list.conversation_id) if task_list.conversation_id else None,
            "run_id": str(task_list.run_id) if task_list.run_id else None,
        }

    def _deserialize_task_list(self, data: dict) -> TaskList:
        return TaskList(
            id=_parse_uuid(data["id"]),
            name=data["name"],
            tasks=[
                Task(
                    id=_parse_uuid(t["id"]),
                    name=t["name"],
                    description=t.get("description", ""),
                    state=TaskState(t["state"]),
                    parent_id=_parse_uuid(t["parent_id"]) if t.get("parent_id") else None,
                    created_at=_parse_datetime(t["created_at"]),
                    updated_at=_parse_datetime(t["updated_at"]),
                    metadata=t.get("metadata", {}),
                )
                for t in data.get("tasks", [])
            ],
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            conversation_id=_parse_uuid(data["conversation_id"]) if data.get("conversation_id") else None,
            run_id=_parse_uuid(data["run_id"]) if data.get("run_id") else None,
        )

    async def save(self, task_list: TaskList, scope: Scope = Scope.PROJECT) -> None:
        path = self._get_task_list_path(task_list.id, scope)
        _ensure_dir(path.parent)
        task_list.updated_at = datetime.utcnow()
        with open(path, "w") as f:
            f.write(_json_dumps(self._serialize_task_list(task_list)))

    async def get(self, task_list_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[TaskList]:
        path = self._get_task_list_path(task_list_id, scope)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return self._deserialize_task_list(data)
        except (json.JSONDecodeError, IOError):
            return None

    async def delete(self, task_list_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        path = self._get_task_list_path(task_list_id, scope)
        if path.exists():
            path.unlink()
            return True
        return False

    async def get_by_conversation(
        self,
        conversation_id: UUID,
        scope: Scope = Scope.PROJECT,
    ) -> Optional[TaskList]:
        tasks_path = self._get_tasks_path(scope)
        if not tasks_path.exists():
            return None

        for file in tasks_path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data.get("conversation_id") == str(conversation_id):
                        return self._deserialize_task_list(data)
            except (json.JSONDecodeError, IOError):
                continue
        return None

    async def update_task(
        self,
        task_list_id: UUID,
        task_id: UUID,
        state: Optional[TaskState] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Scope = Scope.PROJECT,
    ) -> None:
        task_list = await self.get(task_list_id, scope)
        if task_list is None:
            raise ValueError(f"Task list not found: {task_list_id}")

        for task in task_list.tasks:
            if task.id == task_id:
                if state is not None:
                    task.state = state
                if name is not None:
                    task.name = name
                if description is not None:
                    task.description = description
                task.updated_at = datetime.utcnow()
                break
        else:
            raise ValueError(f"Task not found: {task_id}")

        await self.save(task_list, scope)



class FilePreferencesStore(PreferencesStore):
    """
    File-based preferences store.

    Stores preferences in a single JSON file:
    - {base_path}/preferences.json
    """

    def __init__(self, project_dir: Optional[Path] = None):
        self._project_dir = project_dir

    def _get_preferences_path(self, scope: Scope) -> Path:
        return _get_base_path(scope, self._project_dir) / "preferences.json"

    async def _load_preferences(self, scope: Scope) -> dict:
        path = self._get_preferences_path(scope)
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    async def _save_preferences(self, preferences: dict, scope: Scope) -> None:
        path = self._get_preferences_path(scope)
        _ensure_dir(path.parent)
        with open(path, "w") as f:
            f.write(_json_dumps(preferences))

    async def get(self, key: str, scope: Scope = Scope.GLOBAL) -> Optional[Any]:
        preferences = await self._load_preferences(scope)
        return preferences.get(key)

    async def set(self, key: str, value: Any, scope: Scope = Scope.GLOBAL) -> None:
        preferences = await self._load_preferences(scope)
        preferences[key] = value
        await self._save_preferences(preferences, scope)

    async def delete(self, key: str, scope: Scope = Scope.GLOBAL) -> bool:
        preferences = await self._load_preferences(scope)
        if key in preferences:
            del preferences[key]
            await self._save_preferences(preferences, scope)
            return True
        return False

    async def get_all(self, scope: Scope = Scope.GLOBAL) -> dict[str, Any]:
        return await self._load_preferences(scope)


class FileKnowledgeStore(KnowledgeStore):
    """
    File-based knowledge store with optional vector store integration.

    Stores facts and summaries in JSON files:
    - {base_path}/knowledge/facts/{fact_id}.json
    - {base_path}/knowledge/summaries/{summary_id}.json

    Embeddings are stored via an optional VectorStore backend.
    """

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        vector_store: Optional["VectorStore"] = None,
        embedding_client: Optional["EmbeddingClient"] = None,
    ):
        """
        Initialize file-based knowledge store.

        Args:
            project_dir: Base directory for file storage
            vector_store: Optional VectorStore for embedding storage
            embedding_client: Optional EmbeddingClient for generating embeddings
        """
        self._project_dir = project_dir
        self._vector_store = vector_store
        self._embedding_client = embedding_client

    def _get_facts_path(self, scope: Scope) -> Path:
        return _get_base_path(scope, self._project_dir) / "knowledge" / "facts"

    def _get_summaries_path(self, scope: Scope) -> Path:
        return _get_base_path(scope, self._project_dir) / "knowledge" / "summaries"

    def _get_fact_path(self, fact_id: UUID, scope: Scope) -> Path:
        return self._get_facts_path(scope) / f"{fact_id}.json"

    def _get_summary_path(self, summary_id: UUID, scope: Scope) -> Path:
        return self._get_summaries_path(scope) / f"{summary_id}.json"

    def _serialize_fact(self, fact: Fact) -> dict:
        return {
            "id": str(fact.id),
            "key": fact.key,
            "value": fact.value,
            "fact_type": fact.fact_type.value,
            "confidence": fact.confidence,
            "source": fact.source,
            "created_at": fact.created_at.isoformat(),
            "updated_at": fact.updated_at.isoformat(),
            "expires_at": fact.expires_at.isoformat() if fact.expires_at else None,
            "metadata": fact.metadata,
        }

    def _deserialize_fact(self, data: dict) -> Fact:
        return Fact(
            id=_parse_uuid(data["id"]),
            key=data["key"],
            value=data["value"],
            fact_type=FactType(data["fact_type"]),
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            expires_at=_parse_datetime(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )

    def _serialize_summary(self, summary: Summary) -> dict:
        return {
            "id": str(summary.id),
            "content": summary.content,
            "conversation_id": str(summary.conversation_id) if summary.conversation_id else None,
            "conversation_ids": [str(cid) for cid in summary.conversation_ids],
            "start_time": summary.start_time.isoformat() if summary.start_time else None,
            "end_time": summary.end_time.isoformat() if summary.end_time else None,
            "created_at": summary.created_at.isoformat(),
            "metadata": summary.metadata,
        }

    def _deserialize_summary(self, data: dict) -> Summary:
        return Summary(
            id=_parse_uuid(data["id"]),
            content=data["content"],
            conversation_id=_parse_uuid(data["conversation_id"]) if data.get("conversation_id") else None,
            conversation_ids=[_parse_uuid(cid) for cid in data.get("conversation_ids", [])],
            start_time=_parse_datetime(data["start_time"]) if data.get("start_time") else None,
            end_time=_parse_datetime(data["end_time"]) if data.get("end_time") else None,
            created_at=_parse_datetime(data["created_at"]),
            metadata=data.get("metadata", {}),
        )

    # Fact operations
    async def save_fact(self, fact: Fact, scope: Scope = Scope.PROJECT) -> None:
        path = self._get_fact_path(fact.id, scope)
        _ensure_dir(path.parent)
        fact.updated_at = datetime.utcnow()
        with open(path, "w") as f:
            f.write(_json_dumps(self._serialize_fact(fact)))

    async def get_fact(self, fact_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[Fact]:
        path = self._get_fact_path(fact_id, scope)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return self._deserialize_fact(data)
        except (json.JSONDecodeError, IOError):
            return None

    async def get_fact_by_key(self, key: str, scope: Scope = Scope.PROJECT) -> Optional[Fact]:
        facts_path = self._get_facts_path(scope)
        if not facts_path.exists():
            return None
        for file in facts_path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data.get("key") == key:
                        return self._deserialize_fact(data)
            except (json.JSONDecodeError, IOError):
                continue
        return None

    async def list_facts(
        self,
        scope: Scope = Scope.PROJECT,
        fact_type: Optional[FactType] = None,
        limit: int = 100,
    ) -> list[Fact]:
        facts_path = self._get_facts_path(scope)
        if not facts_path.exists():
            return []
        facts = []
        for file in facts_path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    fact = self._deserialize_fact(data)
                    if fact_type is None or fact.fact_type == fact_type:
                        facts.append(fact)
            except (json.JSONDecodeError, IOError):
                continue
        facts.sort(key=lambda f: f.updated_at, reverse=True)
        return facts[:limit]

    async def delete_fact(self, fact_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        path = self._get_fact_path(fact_id, scope)
        if path.exists():
            path.unlink()
            return True
        return False

    # Summary operations
    async def save_summary(self, summary: Summary, scope: Scope = Scope.PROJECT) -> None:
        path = self._get_summary_path(summary.id, scope)
        _ensure_dir(path.parent)
        with open(path, "w") as f:
            f.write(_json_dumps(self._serialize_summary(summary)))

    async def get_summary(self, summary_id: UUID, scope: Scope = Scope.PROJECT) -> Optional[Summary]:
        path = self._get_summary_path(summary_id, scope)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return self._deserialize_summary(data)
        except (json.JSONDecodeError, IOError):
            return None

    async def get_summaries_for_conversation(
        self,
        conversation_id: UUID,
        scope: Scope = Scope.PROJECT,
    ) -> list[Summary]:
        summaries_path = self._get_summaries_path(scope)
        if not summaries_path.exists():
            return []
        summaries = []
        for file in summaries_path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    summary = self._deserialize_summary(data)
                    if (
                        summary.conversation_id == conversation_id
                        or conversation_id in summary.conversation_ids
                    ):
                        summaries.append(summary)
            except (json.JSONDecodeError, IOError):
                continue
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    async def delete_summary(self, summary_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        path = self._get_summary_path(summary_id, scope)
        if path.exists():
            path.unlink()
            return True
        return False

    # Embedding operations (using VectorStore)
    async def save_embedding(self, embedding: Embedding, scope: Scope = Scope.PROJECT) -> None:
        if not self._vector_store:
            raise NotImplementedError("No vector store configured")
        await self._vector_store.add(
            id=str(embedding.id),
            vector=embedding.vector,
            content=embedding.content,
            metadata={
                "content_type": embedding.content_type,
                "source_id": str(embedding.source_id) if embedding.source_id else None,
                "model": embedding.model,
                "dimensions": embedding.dimensions,
                **embedding.metadata,
            },
        )

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        scope: Scope = Scope.PROJECT,
        content_type: Optional[str] = None,
    ) -> list[tuple[Embedding, float]]:
        if not self._vector_store:
            raise NotImplementedError("No vector store configured")

        filter_dict = {"content_type": content_type} if content_type else None
        results = await self._vector_store.search(query_vector, limit=limit, filter=filter_dict)

        return [
            (
                Embedding(
                    id=_parse_uuid(r.id),
                    vector=[],  # Don't return full vector
                    content=r.content,
                    content_type=r.metadata.get("content_type", "text"),
                    source_id=_parse_uuid(r.metadata["source_id"]) if r.metadata.get("source_id") else None,
                    model=r.metadata.get("model"),
                    dimensions=r.metadata.get("dimensions", 0),
                    created_at=datetime.utcnow(),  # Not stored in vector store
                    metadata={k: v for k, v in r.metadata.items()
                              if k not in ("content_type", "source_id", "model", "dimensions")},
                ),
                r.score,
            )
            for r in results
        ]

    async def delete_embedding(self, embedding_id: UUID, scope: Scope = Scope.PROJECT) -> bool:
        if not self._vector_store:
            raise NotImplementedError("No vector store configured")
        return await self._vector_store.delete(str(embedding_id))

    async def close(self) -> None:
        if self._vector_store:
            await self._vector_store.close()


# Type hints for optional imports
try:
    from agent_runtime_core.vectorstore.base import VectorStore
    from agent_runtime_core.vectorstore.embeddings import EmbeddingClient
except ImportError:
    VectorStore = None  # type: ignore
    EmbeddingClient = None  # type: ignore


# =============================================================================
# In-Memory Shared Memory Store (for testing)
# =============================================================================


from agent_runtime_core.persistence.base import SharedMemoryStore, MemoryItem


class InMemorySharedMemoryStore(SharedMemoryStore):
    """
    In-memory implementation of SharedMemoryStore for testing.

    This store keeps all data in memory and is not persistent.
    Useful for unit tests and development.

    Example:
        store = InMemorySharedMemoryStore()

        # Set a memory
        await store.set("user.name", "Chris", scope="user")

        # Get a memory
        item = await store.get("user.name")
        print(item.value)  # "Chris"
    """

    def __init__(self):
        # Storage: dict[composite_key, MemoryItem]
        # composite_key = f"{scope}:{conversation_id or ''}:{system_id or ''}:{key}"
        self._storage: dict[str, MemoryItem] = {}

    def _make_composite_key(
        self,
        key: str,
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
    ) -> str:
        """Create a composite key for storage lookup."""
        scope_part = scope or "user"
        conv_part = str(conversation_id) if conversation_id else ""
        sys_part = system_id or ""
        return f"{scope_part}:{conv_part}:{sys_part}:{key}"

    def _matches_filter(
        self,
        item: MemoryItem,
        prefix: Optional[str] = None,
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> bool:
        """Check if an item matches the given filters."""
        if prefix and not item.key.startswith(prefix):
            return False
        if scope and item.scope != scope:
            return False
        if conversation_id and item.conversation_id != conversation_id:
            return False
        if system_id and item.system_id != system_id:
            return False
        if source and item.source != source:
            return False
        if min_confidence is not None and item.confidence < min_confidence:
            return False
        return True

    async def get(
        self,
        key: str,
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """Get a memory item by key."""
        composite_key = self._make_composite_key(key, scope, conversation_id, system_id)
        return self._storage.get(composite_key)

    async def set(
        self,
        key: str,
        value: Any,
        scope: str = "user",
        source: str = "agent",
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
        expires_at: Optional[datetime] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
    ) -> MemoryItem:
        """Set a memory item. Creates or updates."""
        from uuid import uuid4

        composite_key = self._make_composite_key(key, scope, conversation_id, system_id)
        existing = self._storage.get(composite_key)

        now = datetime.utcnow()
        if existing:
            # Update existing
            item = MemoryItem(
                id=existing.id,
                key=key,
                value=value,
                scope=scope,
                created_at=existing.created_at,
                updated_at=now,
                source=source,
                confidence=confidence,
                metadata=metadata or {},
                expires_at=expires_at,
                conversation_id=conversation_id,
                system_id=system_id,
            )
        else:
            # Create new
            item = MemoryItem(
                id=uuid4(),
                key=key,
                value=value,
                scope=scope,
                created_at=now,
                updated_at=now,
                source=source,
                confidence=confidence,
                metadata=metadata or {},
                expires_at=expires_at,
                conversation_id=conversation_id,
                system_id=system_id,
            )

        self._storage[composite_key] = item
        return item

    async def delete(
        self,
        key: str,
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
    ) -> bool:
        """Delete a memory item."""
        composite_key = self._make_composite_key(key, scope, conversation_id, system_id)
        if composite_key in self._storage:
            del self._storage[composite_key]
            return True
        return False

    async def list(
        self,
        prefix: Optional[str] = None,
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """List memory items with optional filters."""
        results = []
        for item in self._storage.values():
            if self._matches_filter(
                item, prefix, scope, conversation_id, system_id, source, min_confidence
            ):
                results.append(item)

        # Sort by updated_at descending
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results[:limit]

    async def get_many(
        self,
        keys: List[str],
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
    ) -> Dict[str, MemoryItem]:
        """Get multiple memory items by keys."""
        results = {}
        for key in keys:
            item = await self.get(key, scope, conversation_id, system_id)
            if item:
                results[key] = item
        return results

    async def set_many(
        self,
        items: List[tuple],
        scope: str = "user",
        source: str = "agent",
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        """Set multiple memory items atomically."""
        results = []
        for key, value in items:
            item = await self.set(
                key, value, scope, source,
                conversation_id=conversation_id,
                system_id=system_id,
            )
            results.append(item)
        return results

    async def clear(
        self,
        scope: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        system_id: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> int:
        """Clear memory items."""
        to_delete = []
        for composite_key, item in self._storage.items():
            if self._matches_filter(item, prefix, scope, conversation_id, system_id):
                to_delete.append(composite_key)

        for key in to_delete:
            del self._storage[key]

        return len(to_delete)