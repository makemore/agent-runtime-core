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

__version__ = "0.5.2"

# Core interfaces
from agent_runtime_core.interfaces import (
    AgentRuntime,
    EventType,
    ErrorInfo,
    LLMClient,
    LLMResponse,
    LLMStreamChunk,
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
    # Manager
    PersistenceManager,
    PersistenceConfig,
    get_persistence_manager,
    configure_persistence,
)

__all__ = [
    # Version
    "__version__",
    # Interfaces
    "AgentRuntime",
    "LLMClient",
    "LLMResponse",
    "LLMStreamChunk",
    "Message",
    "RunContext",
    "RunResult",
    "ToolRegistry",
    "Tool",
    "ToolDefinition",
    "TraceSink",
    "EventType",
    "ErrorInfo",
        "ToolCallingAgent",
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
    # Persistence - Manager
    "PersistenceManager",
    "PersistenceConfig",
    "get_persistence_manager",
    "configure_persistence",
]
