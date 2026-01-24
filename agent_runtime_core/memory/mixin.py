"""
Memory mixin for adding cross-conversation memory to agents.

This mixin can be combined with ToolCallingAgent or any AgentRuntime
to add automatic memory extraction and recall.
"""

import logging
from typing import Any, Optional

from agent_runtime_core.interfaces import LLMClient, RunContext, RunResult
from agent_runtime_core.persistence.base import KnowledgeStore, FactType
from agent_runtime_core.memory.manager import (
    MemoryManager,
    MemoryConfig,
    RecalledMemory,
)

logger = logging.getLogger(__name__)


class MemoryEnabledAgent:
    """
    Mixin that adds cross-conversation memory to agents.
    
    Add this mixin to your agent class to enable automatic memory
    extraction and recall. The mixin hooks into the agent lifecycle
    to:
    
    1. Recall relevant memories before each run
    2. Inject memories into the system prompt
    3. Extract new memories after each run
    
    Example:
        from agent_runtime_core import ToolCallingAgent, ToolRegistry
        from agent_runtime_core.memory import MemoryEnabledAgent
        
        class MyAgent(MemoryEnabledAgent, ToolCallingAgent):
            memory_enabled = True
            
            @property
            def key(self) -> str:
                return "my-agent"
            
            @property
            def system_prompt(self) -> str:
                return "You are a helpful assistant."
            
            @property
            def tools(self) -> ToolRegistry:
                return ToolRegistry()
    
    Configuration:
        - memory_enabled: Set to True to enable memory (default: False)
        - memory_config: Optional MemoryConfig instance
        - memory_prompt_style: How to format memories ("list", "prose", "structured")
    """
    
    # Class-level configuration (can be overridden in subclasses)
    memory_enabled: bool = False
    memory_config: Optional[MemoryConfig] = None
    memory_prompt_style: str = "list"
    
    # Instance state (set during initialization)
    _memory_manager: Optional[MemoryManager] = None
    _recalled_memories: list[RecalledMemory] = []
    
    def configure_memory(
        self,
        knowledge_store: KnowledgeStore,
        llm_client: Optional[LLMClient] = None,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        """
        Configure the memory system for this agent.
        
        Call this before running the agent if memory_enabled is True.
        
        Args:
            knowledge_store: Store for persisting memories
            llm_client: LLM client for extraction (uses agent's client if not provided)
            config: Memory configuration
        """
        if llm_client is None:
            # Try to get from agent's get_llm_client method
            if hasattr(self, 'get_llm_client'):
                llm_client = self.get_llm_client()
            else:
                raise ValueError("llm_client required when agent doesn't have get_llm_client")
        
        self._memory_manager = MemoryManager(
            knowledge_store=knowledge_store,
            llm_client=llm_client,
            config=config or self.memory_config or MemoryConfig(),
        )
    
    def get_memory_manager(self) -> Optional[MemoryManager]:
        """Get the memory manager instance."""
        return self._memory_manager
    
    async def recall_memories_for_context(
        self,
        ctx: RunContext,
    ) -> list[RecalledMemory]:
        """
        Recall relevant memories for the current context.
        
        Override this method to customize memory recall logic.
        
        Args:
            ctx: The run context
            
        Returns:
            List of relevant memories
        """
        if not self._memory_manager:
            return []
        
        # Get user_id from context metadata if available
        user_id = ctx.metadata.get("user_id") or ctx.metadata.get("user")
        
        # Recall memories based on input messages
        return await self._memory_manager.recall_memories(
            messages=ctx.input_messages,
            user_id=str(user_id) if user_id else None,
        )
    
    def format_memories_for_prompt(
        self,
        memories: list[RecalledMemory],
    ) -> str:
        """
        Format memories for inclusion in the system prompt.
        
        Override this method to customize memory formatting.
        
        Args:
            memories: List of recalled memories
            
        Returns:
            Formatted string to append to system prompt
        """
        if not self._memory_manager or not memories:
            return ""
        
        return self._memory_manager.format_memories_for_prompt(
            memories,
            format_style=self.memory_prompt_style,
        )
    
    async def extract_memories_from_run(
        self,
        ctx: RunContext,
        result: RunResult,
    ) -> None:
        """
        Extract memories from a completed run.
        
        Override this method to customize memory extraction logic.
        
        Args:
            ctx: The run context
            result: The run result
        """
        if not self._memory_manager:
            return
        
        # Get user_id from context metadata
        user_id = ctx.metadata.get("user_id") or ctx.metadata.get("user")
        
        # Combine input messages with result messages for extraction
        all_messages = list(ctx.input_messages)
        if result.final_messages:
            all_messages.extend(result.final_messages)
        
        # Extract memories
        await self._memory_manager.extract_memories(
            messages=all_messages,
            user_id=str(user_id) if user_id else None,
        )
    
    # ==========================================================================
    # Hooks for ToolCallingAgent integration
    # ==========================================================================
    
    async def before_run(self, ctx: RunContext) -> None:
        """
        Hook called before the agent run starts.
        
        Recalls relevant memories and stores them for prompt injection.
        """
        # Call parent's before_run if it exists
        if hasattr(super(), 'before_run'):
            await super().before_run(ctx)
        
        if self.memory_enabled and self._memory_manager:
            self._recalled_memories = await self.recall_memories_for_context(ctx)
            if self._recalled_memories:
                logger.info(f"Recalled {len(self._recalled_memories)} memories for run")
    
    async def after_run(self, ctx: RunContext, result: RunResult) -> RunResult:
        """
        Hook called after the agent run completes.
        
        Extracts memories from the conversation.
        """
        # Call parent's after_run if it exists
        if hasattr(super(), 'after_run'):
            result = await super().after_run(ctx, result)
        
        if self.memory_enabled and self._memory_manager:
            await self.extract_memories_from_run(ctx, result)
        
        return result
    
    @property
    def system_prompt_with_memory(self) -> str:
        """
        Get the system prompt with memories injected.
        
        This is used internally by the mixin. Override system_prompt
        in your agent class, not this property.
        """
        base_prompt = self.system_prompt if hasattr(self, 'system_prompt') else ""
        
        if not self.memory_enabled or not self._recalled_memories:
            return base_prompt
        
        memory_text = self.format_memories_for_prompt(self._recalled_memories)
        if memory_text:
            return f"{base_prompt}\n\n{memory_text}"
        
        return base_prompt


# =============================================================================
# Convenience function for creating memory-enabled agents
# =============================================================================


def with_memory(
    agent_class: type,
    knowledge_store: KnowledgeStore,
    llm_client: Optional[LLMClient] = None,
    config: Optional[MemoryConfig] = None,
) -> type:
    """
    Create a memory-enabled version of an agent class.
    
    This is a convenience function for adding memory to existing agents
    without modifying their class definition.
    
    Example:
        from agent_runtime_core import ToolCallingAgent
        from agent_runtime_core.memory import with_memory
        
        # Original agent
        class MyAgent(ToolCallingAgent):
            ...
        
        # Memory-enabled version
        MemoryMyAgent = with_memory(
            MyAgent,
            knowledge_store=FileKnowledgeStore(),
        )
        
        agent = MemoryMyAgent()
    
    Args:
        agent_class: The agent class to enhance
        knowledge_store: Store for persisting memories
        llm_client: Optional LLM client for extraction
        config: Optional memory configuration
        
    Returns:
        A new class with memory capabilities
    """
    class MemoryEnabledVersion(MemoryEnabledAgent, agent_class):
        memory_enabled = True
        memory_config = config
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.configure_memory(
                knowledge_store=knowledge_store,
                llm_client=llm_client,
                config=config,
            )
    
    MemoryEnabledVersion.__name__ = f"MemoryEnabled{agent_class.__name__}"
    MemoryEnabledVersion.__qualname__ = f"MemoryEnabled{agent_class.__qualname__}"
    
    return MemoryEnabledVersion

