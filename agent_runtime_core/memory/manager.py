"""
Memory manager for cross-conversation memory.

Handles extraction of memories from conversations and recall of relevant
memories for new conversations.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from agent_runtime_core.interfaces import LLMClient, Message
from agent_runtime_core.persistence.base import (
    Fact,
    FactType,
    KnowledgeStore,
    Scope,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MemoryConfig:
    """Configuration for memory extraction and recall."""
    
    # Extraction settings
    extract_after_messages: int = 4  # Extract after this many messages
    extract_on_conversation_end: bool = True  # Always extract at end
    max_facts_per_extraction: int = 5  # Max facts to extract at once
    
    # Recall settings
    max_memories_to_recall: int = 10  # Max memories to include in context
    relevance_threshold: float = 0.5  # Min relevance score (0-1)
    
    # What to extract
    extract_user_facts: bool = True  # Name, preferences, etc.
    extract_project_facts: bool = True  # Project-specific info
    extract_preferences: bool = True  # User preferences
    
    # Storage
    scope: Scope = Scope.GLOBAL  # Where to store memories
    
    # Model settings (optional override)
    extraction_model: Optional[str] = None  # Model for extraction
    

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExtractedMemory:
    """A memory extracted from a conversation."""
    
    key: str  # Unique identifier (e.g., "user_name", "preferred_language")
    value: Any  # The memory content
    fact_type: FactType  # Type of fact
    confidence: float = 1.0  # How confident we are (0-1)
    source_message: Optional[str] = None  # The message it came from


@dataclass
class RecalledMemory:
    """A memory recalled for use in a conversation."""
    
    key: str
    value: Any
    fact_type: FactType
    relevance: float = 1.0  # How relevant to current context (0-1)
    created_at: Optional[datetime] = None


# =============================================================================
# Prompts
# =============================================================================


EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system. Your job is to identify important facts, preferences, and information from conversations that should be remembered for future interactions.

Extract ONLY information that would be useful to remember across conversations, such as:
- User's name, role, or identity
- User preferences (communication style, formatting, tools, etc.)
- Important project details (tech stack, constraints, goals)
- Recurring topics or interests
- Explicit requests to remember something

Do NOT extract:
- Temporary or session-specific information
- Information that changes frequently
- Sensitive information (passwords, API keys, etc.)
- Generic conversation content

For each fact, provide:
- key: A unique, descriptive identifier (snake_case, e.g., "user_name", "preferred_language")
- value: The actual information (can be string, number, list, or object)
- type: One of "user", "project", "preference", "context"
- confidence: How confident you are this is correct (0.0 to 1.0)

Respond with a JSON array of facts. If no facts should be extracted, respond with an empty array [].
"""

EXTRACTION_USER_PROMPT = """Extract memorable facts from this conversation:

{conversation}

Respond with a JSON array of facts to remember. Example format:
[
  {{"key": "user_name", "value": "Alice", "type": "user", "confidence": 1.0}},
  {{"key": "preferred_theme", "value": "dark", "type": "preference", "confidence": 0.9}}
]

If nothing should be remembered, respond with: []"""


RECALL_SYSTEM_PROMPT = """You are a memory relevance system. Given a set of stored memories and a current conversation context, determine which memories are relevant.

For each memory, assign a relevance score from 0.0 to 1.0:
- 1.0: Directly relevant and should definitely be used
- 0.7-0.9: Likely relevant, good to include
- 0.4-0.6: Possibly relevant, include if space allows
- 0.1-0.3: Tangentially related
- 0.0: Not relevant at all

Respond with a JSON array of objects with "key" and "relevance" fields.
"""

RECALL_USER_PROMPT = """Current conversation context:
{context}

Available memories:
{memories}

Which memories are relevant? Respond with JSON array:
[{{"key": "memory_key", "relevance": 0.9}}, ...]"""


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """
    Manages cross-conversation memory extraction and recall.
    
    Uses an LLM to extract memorable facts from conversations and
    recall relevant memories for new conversations.
    """
    
    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        llm_client: LLMClient,
        config: Optional[MemoryConfig] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize the memory manager.
        
        Args:
            knowledge_store: Store for persisting memories (facts)
            llm_client: LLM client for extraction/recall
            config: Memory configuration
            user_id: Optional user ID for scoping memories
        """
        self._store = knowledge_store
        self._llm = llm_client
        self._config = config or MemoryConfig()
        self._user_id = user_id

    def _get_memory_key(self, key: str) -> str:
        """Get the full key including user_id prefix if set."""
        if self._user_id:
            return f"user:{self._user_id}:{key}"
        return key

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format messages for the extraction prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _parse_fact_type(self, type_str: str) -> FactType:
        """Parse a fact type string to FactType enum."""
        type_map = {
            "user": FactType.USER,
            "project": FactType.PROJECT,
            "preference": FactType.PREFERENCE,
            "context": FactType.CONTEXT,
        }
        return type_map.get(type_str.lower(), FactType.CUSTOM)

    async def extract_memories(
        self,
        messages: list[Message],
        user_id: Optional[str] = None,
        save: bool = True,
    ) -> list[ExtractedMemory]:
        """
        Extract memorable facts from a conversation.

        Args:
            messages: Conversation messages to analyze
            user_id: Optional user ID (overrides instance user_id)
            save: Whether to save extracted memories to store

        Returns:
            List of extracted memories
        """
        if not messages:
            return []

        effective_user_id = user_id or self._user_id
        conversation_text = self._format_conversation(messages)

        # Call LLM to extract facts
        try:
            response = await self._llm.generate(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": EXTRACTION_USER_PROMPT.format(
                        conversation=conversation_text
                    )},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
            )

            # Parse response
            content = response.message.get("content", "[]")
            # Handle potential markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            facts_data = json.loads(content.strip())

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to extract memories: {e}")
            return []

        # Convert to ExtractedMemory objects
        extracted = []
        for fact_data in facts_data[:self._config.max_facts_per_extraction]:
            memory = ExtractedMemory(
                key=fact_data.get("key", ""),
                value=fact_data.get("value"),
                fact_type=self._parse_fact_type(fact_data.get("type", "custom")),
                confidence=float(fact_data.get("confidence", 1.0)),
            )
            if memory.key:  # Only include if key is set
                extracted.append(memory)

        # Save to store if requested
        if save and extracted:
            for memory in extracted:
                await self._save_memory(memory, effective_user_id)

        logger.info(f"Extracted {len(extracted)} memories from conversation")
        return extracted

    async def _save_memory(
        self,
        memory: ExtractedMemory,
        user_id: Optional[str] = None,
    ) -> None:
        """Save an extracted memory to the knowledge store."""
        effective_user_id = user_id or self._user_id
        full_key = self._get_memory_key(memory.key) if effective_user_id else memory.key

        # Check if fact already exists
        existing = await self._store.get_fact_by_key(full_key, self._config.scope)

        if existing:
            # Update existing fact
            existing.value = memory.value
            existing.confidence = memory.confidence
            existing.updated_at = datetime.utcnow()
            await self._store.save_fact(existing, self._config.scope)
        else:
            # Create new fact
            fact = Fact(
                id=uuid4(),
                key=full_key,
                value=memory.value,
                fact_type=memory.fact_type,
                confidence=memory.confidence,
                source=f"user:{effective_user_id}" if effective_user_id else None,
                metadata={"user_id": effective_user_id} if effective_user_id else {},
            )
            await self._store.save_fact(fact, self._config.scope)

    async def recall_memories(
        self,
        query: Optional[str] = None,
        messages: Optional[list[Message]] = None,
        user_id: Optional[str] = None,
        max_memories: Optional[int] = None,
    ) -> list[RecalledMemory]:
        """
        Recall relevant memories for a conversation.

        Args:
            query: Optional query to find relevant memories
            messages: Optional conversation context
            user_id: Optional user ID (overrides instance user_id)
            max_memories: Max memories to return (overrides config)

        Returns:
            List of relevant memories
        """
        effective_user_id = user_id or self._user_id
        max_count = max_memories or self._config.max_memories_to_recall

        # Get all facts for this user
        all_facts = await self._store.list_facts(
            scope=self._config.scope,
            limit=100,  # Get more than we need for filtering
        )

        # Filter to user's facts if user_id is set
        if effective_user_id:
            prefix = f"user:{effective_user_id}:"
            all_facts = [f for f in all_facts if f.key.startswith(prefix)]

        if not all_facts:
            return []

        # If no query/messages, return all memories (up to limit)
        if not query and not messages:
            return [
                RecalledMemory(
                    key=self._strip_user_prefix(f.key, effective_user_id),
                    value=f.value,
                    fact_type=f.fact_type,
                    relevance=1.0,
                    created_at=f.created_at,
                )
                for f in all_facts[:max_count]
            ]

        # Use LLM to rank relevance
        context = query or self._format_conversation(messages or [])
        memories_text = "\n".join(
            f"- {self._strip_user_prefix(f.key, effective_user_id)}: {f.value}"
            for f in all_facts
        )

        try:
            response = await self._llm.generate(
                messages=[
                    {"role": "system", "content": RECALL_SYSTEM_PROMPT},
                    {"role": "user", "content": RECALL_USER_PROMPT.format(
                        context=context,
                        memories=memories_text,
                    )},
                ],
                temperature=0.1,
            )

            content = response.message.get("content", "[]")
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            relevance_data = json.loads(content.strip())
            relevance_map = {r["key"]: r["relevance"] for r in relevance_data}

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to rank memories: {e}")
            # Fall back to returning all memories
            relevance_map = {
                self._strip_user_prefix(f.key, effective_user_id): 1.0
                for f in all_facts
            }

        # Build recalled memories with relevance scores
        recalled = []
        for fact in all_facts:
            stripped_key = self._strip_user_prefix(fact.key, effective_user_id)
            relevance = relevance_map.get(stripped_key, 0.0)

            if relevance >= self._config.relevance_threshold:
                recalled.append(RecalledMemory(
                    key=stripped_key,
                    value=fact.value,
                    fact_type=fact.fact_type,
                    relevance=relevance,
                    created_at=fact.created_at,
                ))

        # Sort by relevance and limit
        recalled.sort(key=lambda m: m.relevance, reverse=True)
        return recalled[:max_count]

    def _strip_user_prefix(self, key: str, user_id: Optional[str]) -> str:
        """Strip the user prefix from a key."""
        if user_id:
            prefix = f"user:{user_id}:"
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    async def get_memory(
        self,
        key: str,
        user_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get a specific memory by key.

        Args:
            key: Memory key
            user_id: Optional user ID

        Returns:
            Memory value or None
        """
        effective_user_id = user_id or self._user_id
        full_key = self._get_memory_key(key) if effective_user_id else key

        fact = await self._store.get_fact_by_key(full_key, self._config.scope)
        return fact.value if fact else None

    async def set_memory(
        self,
        key: str,
        value: Any,
        fact_type: FactType = FactType.CUSTOM,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Manually set a memory.

        Args:
            key: Memory key
            value: Memory value
            fact_type: Type of fact
            user_id: Optional user ID
        """
        memory = ExtractedMemory(
            key=key,
            value=value,
            fact_type=fact_type,
            confidence=1.0,
        )
        await self._save_memory(memory, user_id or self._user_id)

    async def delete_memory(
        self,
        key: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a specific memory.

        Args:
            key: Memory key
            user_id: Optional user ID

        Returns:
            True if memory existed and was deleted
        """
        effective_user_id = user_id or self._user_id
        full_key = self._get_memory_key(key) if effective_user_id else key

        fact = await self._store.get_fact_by_key(full_key, self._config.scope)
        if fact:
            return await self._store.delete_fact(fact.id, self._config.scope)
        return False

    async def clear_memories(
        self,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Clear all memories for a user.

        Args:
            user_id: Optional user ID

        Returns:
            Number of memories deleted
        """
        effective_user_id = user_id or self._user_id

        all_facts = await self._store.list_facts(scope=self._config.scope, limit=1000)

        if effective_user_id:
            prefix = f"user:{effective_user_id}:"
            facts_to_delete = [f for f in all_facts if f.key.startswith(prefix)]
        else:
            facts_to_delete = all_facts

        count = 0
        for fact in facts_to_delete:
            if await self._store.delete_fact(fact.id, self._config.scope):
                count += 1

        return count

    def format_memories_for_prompt(
        self,
        memories: list[RecalledMemory],
        format_style: str = "list",
    ) -> str:
        """
        Format recalled memories for inclusion in a prompt.

        Args:
            memories: List of recalled memories
            format_style: "list", "prose", or "structured"

        Returns:
            Formatted string for prompt inclusion
        """
        if not memories:
            return ""

        if format_style == "list":
            lines = ["Remembered information about the user:"]
            for m in memories:
                lines.append(f"- {m.key}: {m.value}")
            return "\n".join(lines)

        elif format_style == "prose":
            facts = [f"{m.key} is {m.value}" for m in memories]
            return "What I remember: " + "; ".join(facts) + "."

        elif format_style == "structured":
            data = {m.key: m.value for m in memories}
            return f"User context: {json.dumps(data)}"

        return ""

