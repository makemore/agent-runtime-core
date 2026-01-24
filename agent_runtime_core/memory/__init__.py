"""
Cross-conversation memory system for AI agents.

This module provides automatic memory extraction and recall across conversations,
allowing agents to remember facts, preferences, and context about users.

Key Components:
- MemoryManager: Extracts and recalls memories using LLM
- MemoryConfig: Configuration for memory behavior
- MemoryEnabledAgent: Mixin for adding memory to agents

Example usage:
    from agent_runtime_core.memory import MemoryManager, MemoryConfig
    from agent_runtime_core.persistence import FileKnowledgeStore
    from agent_runtime_core.llm import get_llm_client
    
    # Setup memory manager
    knowledge_store = FileKnowledgeStore()
    llm = get_llm_client()
    memory = MemoryManager(knowledge_store, llm)
    
    # Extract memories from a conversation
    messages = [
        {"role": "user", "content": "My name is Alice and I prefer dark mode."},
        {"role": "assistant", "content": "Nice to meet you, Alice! I've noted your preference for dark mode."},
    ]
    await memory.extract_memories(messages, user_id="user-123")
    
    # Recall memories for a new conversation
    relevant_memories = await memory.recall_memories(
        query="What theme does the user prefer?",
        user_id="user-123",
    )
    
    # Or use with ToolCallingAgent via mixin
    from agent_runtime_core.memory import MemoryEnabledAgent
    from agent_runtime_core import ToolCallingAgent
    
    class MyAgent(MemoryEnabledAgent, ToolCallingAgent):
        memory_enabled = True  # Enable memory for this agent
        
        @property
        def key(self) -> str:
            return "my-agent"
        
        @property
        def system_prompt(self) -> str:
            return "You are a helpful assistant."
        
        @property
        def tools(self) -> ToolRegistry:
            return ToolRegistry()
"""

from agent_runtime_core.memory.manager import (
    MemoryManager,
    MemoryConfig,
    ExtractedMemory,
    RecalledMemory,
)
from agent_runtime_core.memory.mixin import MemoryEnabledAgent

__all__ = [
    "MemoryManager",
    "MemoryConfig",
    "ExtractedMemory",
    "RecalledMemory",
    "MemoryEnabledAgent",
]

