"""
agent_runtime - A standalone Python package for building AI agent systems.

This package provides:
- Core interfaces for agent runtimes
- Queue, event bus, and state store implementations
- LLM client abstractions
- Tracing and observability
- A runner for executing agent runs

Example usage:
    from agent_runtime_core import (
        AgentRuntime,
        RunContext,
        RunResult,
        Tool,
        configure,
    )
    
    # Configure the runtime
    configure(
        model_provider="openai",
        queue_backend="memory",
    )
    
    # Create a custom agent runtime
    class MyAgent(AgentRuntime):
        @property
        def key(self) -> str:
            return "my-agent"
        
        async def run(self, ctx: RunContext) -> RunResult:
            # Your agent logic here
            return RunResult(final_output={"message": "Hello!"})
"""

__version__ = "0.7.0"

# Core interfaces
from agent_runtime_core.interfaces import (
    AgentRuntime,
    EventType,
    EventVisibility,
    ErrorInfo,
    LLMClient,
    LLMResponse,
    LLMStreamChunk,
    LLMToolCall,
    Message,
    RunContext,
    RunResult,
    Tool,
    ToolDefinition,
    ToolRegistry,
    TraceSink,
)


# Tool Calling Agent base class
from agent_runtime_core.tool_calling_agent import ToolCallingAgent

# Agentic loop helper
from agent_runtime_core.agentic_loop import (
    run_agentic_loop,
    AgenticLoopResult,
)

# Configuration
from agent_runtime_core.config import (
    RuntimeConfig,
    configure,
    get_config,
)

# Registry
from agent_runtime_core.registry import (
    register_runtime,
    get_runtime,
    list_runtimes,
    unregister_runtime,
    clear_registry,
)

# Runner
from agent_runtime_core.runner import (
    AgentRunner,
    RunnerConfig,
    RunContextImpl,
)

# Step execution for long-running multi-step agents
from agent_runtime_core.steps import (
    Step,
    StepExecutor,
    StepResult,
    StepStatus,
    ExecutionState,
    StepExecutionError,
    StepCancelledError,
)

# Concrete RunContext implementations for different use cases
from agent_runtime_core.contexts import (
    InMemoryRunContext,
    FileRunContext,
)

# Testing utilities
from agent_runtime_core.testing import (
    MockRunContext,
    MockLLMClient,
    MockLLMResponse,
    LLMEvaluator,
    create_test_context,
    run_agent_test,
)

# Persistence (memory, conversations, tasks, preferences)
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
    ToolResult,
    TaskList,
    Task,
    TaskState,
    # File implementations
    FileMemoryStore,
    FileConversationStore,
    FileTaskStore,
    FilePreferencesStore,
    FileKnowledgeStore,
    # Manager
    PersistenceManager,
    PersistenceConfig,
    get_persistence_manager,
    configure_persistence,
)

# Agent configuration schema (portable JSON format)
from agent_runtime_core.config_schema import (
    AgentConfig,
    ToolConfig,
    KnowledgeConfig,
    SubAgentToolConfig,
)

# JSON-based runtime (loads from AgentConfig)
from agent_runtime_core.json_runtime import (
    JsonAgentRuntime,
    ConfiguredTool,
    SubAgentTool,
    resolve_function,
)

# Vector store (optional - requires additional dependencies)
# Import these directly from agent_runtime_core.vectorstore when needed:
#   from agent_runtime_core.vectorstore import (
#       VectorStore, VectorRecord, VectorSearchResult,
#       EmbeddingClient, OpenAIEmbeddings, VertexAIEmbeddings,
#       get_vector_store, get_embedding_client,
#   )

# RAG (Retrieval Augmented Generation)
from agent_runtime_core.rag import (
    chunk_text,
    ChunkingConfig,
    TextChunk,
)

# RAG services are imported lazily to avoid circular dependencies
# Import directly when needed:
#   from agent_runtime_core.rag import KnowledgeIndexer, KnowledgeRetriever

# Tool schema builder utilities
from agent_runtime_core.tools import (
    ToolSchema,
    ToolSchemaBuilder,
    ToolParameter,
    schemas_to_openai_format,
)

# Multi-agent support (agent-as-tool pattern)
from agent_runtime_core.multi_agent import (
    AgentTool,
    AgentInvocationResult,
    InvocationMode,
    ContextMode,
    SubAgentContext,
    invoke_agent,
    create_agent_tool_handler,
    register_agent_tools,
    build_sub_agent_messages,
)

# Cross-conversation memory
# Import directly when needed for full functionality:
#   from agent_runtime_core.memory import (
#       MemoryManager, MemoryConfig, MemoryEnabledAgent,
#       ExtractedMemory, RecalledMemory, with_memory,
#   )

__all__ = [
    # Version
    "__version__",
    # Interfaces
    "AgentRuntime",
    "LLMClient",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMToolCall",
    "Message",
    "RunContext",
    "RunResult",
    "ToolRegistry",
    "Tool",
    "ToolDefinition",
    "TraceSink",
    "EventType",
    "EventVisibility",
    "ErrorInfo",
    # Tool calling
    "ToolCallingAgent",
    "run_agentic_loop",
    "AgenticLoopResult",
    # Configuration
    "RuntimeConfig",
    "configure",
    "get_config",
    # Registry
    "register_runtime",
    "get_runtime",
    "list_runtimes",
    "unregister_runtime",
    "clear_registry",
    # Runner
    "AgentRunner",
    "RunnerConfig",
    "RunContextImpl",
    # Step execution
    "Step",
    "StepExecutor",
    "StepResult",
    "StepStatus",
    "ExecutionState",
    "StepExecutionError",
    "StepCancelledError",
    # Concrete RunContext implementations
    "InMemoryRunContext",
    "FileRunContext",
    # Testing
    "MockRunContext",
    "MockLLMClient",
    "MockLLMResponse",
    "LLMEvaluator",
    "create_test_context",
    "run_agent_test",
    # Persistence - Abstract interfaces
    "MemoryStore",
    "ConversationStore",
    "TaskStore",
    "PreferencesStore",
    "Scope",
    # Persistence - Data classes
    "Conversation",
    "ConversationMessage",
    "ToolCall",
    "ToolResult",
    "TaskList",
    "Task",
    "TaskState",
    # Persistence - File implementations
    "FileMemoryStore",
    "FileConversationStore",
    "FileTaskStore",
    "FilePreferencesStore",
    "FileKnowledgeStore",
    # Persistence - Manager
    "PersistenceManager",
    "PersistenceConfig",
    "get_persistence_manager",
    "configure_persistence",
    # Agent configuration schema
    "AgentConfig",
    "ToolConfig",
    "KnowledgeConfig",
    "SubAgentToolConfig",
    # JSON-based runtime
    "JsonAgentRuntime",
    "ConfiguredTool",
    "SubAgentTool",
    "resolve_function",
    # RAG (Retrieval Augmented Generation)
    "chunk_text",
    "ChunkingConfig",
    "TextChunk",
    # Tool schema builder
    "ToolSchema",
    "ToolSchemaBuilder",
    "ToolParameter",
    "schemas_to_openai_format",
    # Multi-agent support
    "AgentTool",
    "AgentInvocationResult",
    "InvocationMode",
    "ContextMode",
    "SubAgentContext",
    "invoke_agent",
    "create_agent_tool_handler",
    "register_agent_tools",
    "build_sub_agent_messages",
]
