"""
agent_runtime - A standalone Python package for building AI agent systems.

This package provides:
- Core interfaces for agent runtimes
- Queue, event bus, and state store implementations
- LLM client abstractions
- Tracing and observability
- A runner for executing agent runs

Example usage:
    from agent_runtime import (
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

__version__ = "0.1.5"

# Core interfaces
from agent_runtime.interfaces import (
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

# Configuration
from agent_runtime.config import (
    RuntimeConfig,
    configure,
    get_config,
)

# Registry
from agent_runtime.registry import (
    register_runtime,
    get_runtime,
    list_runtimes,
    unregister_runtime,
    clear_registry,
)

# Runner
from agent_runtime.runner import (
    AgentRunner,
    RunnerConfig,
    RunContextImpl,
)


# Testing utilities
from agent_runtime.testing import (
    MockRunContext,
    MockLLMClient,
    MockLLMResponse,
    LLMEvaluator,
    create_test_context,
    run_agent_test,
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
    # Testing
    "MockRunContext",
    "MockLLMClient",
    "MockLLMResponse",
    "LLMEvaluator",
    "create_test_context",
    "run_agent_test",
]
