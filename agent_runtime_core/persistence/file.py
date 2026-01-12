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
from typing import Any, Optional
from uuid import UUID

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
