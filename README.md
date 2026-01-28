# agent-runtime-core

[![PyPI version](https://badge.fury.io/py/agent-runtime-core.svg)](https://badge.fury.io/py/agent-runtime-core)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, framework-agnostic Python library for building AI agent systems. Provides the core abstractions and implementations needed to build production-ready AI agents without tying you to any specific framework.

## Recent Updates

| Version | Date | Changes |
|---------|------|---------|
| **0.8.0** | 2026-01-28 | **File Ingestion** - Pluggable file processors (PDF, images, DOCX, XLSX, CSV), OCR providers (Tesseract, Google Vision, AWS Textract, Azure), AI vision (OpenAI, Anthropic, Gemini), file read/write tools |
| **0.7.1** | 2026-01-24 | RAG module, vector stores (sqlite-vec, Vertex AI), memory system, multi-agent support, agentic loop, JSON runtime |
| **0.6.0** | 2025-01-23 | Enhanced registry with factory functions and class registration |
| **0.5.2** | 2025-01-14 | Add ToolCallingAgent base class, execute_with_events helper |
| **0.5.1** | 2025-01-13 | Bug fixes and improvements |
| **0.5.0** | 2025-01-12 | Initial stable release |

## Features
- 🔌 **Framework Agnostic** - Works with LangGraph, CrewAI, OpenAI Agents, or your own custom loops
- 🤖 **Model Agnostic** - OpenAI, Anthropic, or any provider via LiteLLM
- 📦 **Zero Required Dependencies** - Core library has no dependencies; add only what you need
- 🔄 **Async First** - Built for modern async Python with full sync support
- 🛠️ **Pluggable Backends** - Memory, Redis, or SQLite for queues, events, and state
- 📊 **Observable** - Built-in tracing with optional Langfuse integration
- 🧩 **Composable** - Mix and match components to build your ideal agent system

## Installation

```bash
# Core library (no dependencies)
pip install agent-runtime-core

# With specific LLM providers
pip install agent-runtime-core[openai]
pip install agent-runtime-core[anthropic]
pip install agent-runtime-core[litellm]

# With Redis backend support
pip install agent-runtime-core[redis]

# With observability
pip install agent-runtime-core[langfuse]

# Everything
pip install agent-runtime-core[all]
```

## Quick Start

### Basic Configuration

```python
from agent_runtime_core import configure, get_config

# Configure the runtime
configure(
    model_provider="openai",
    openai_api_key="sk-...",  # Or use OPENAI_API_KEY env var
    default_model="gpt-4o",
)

# Access configuration anywhere
config = get_config()
print(config.model_provider)  # "openai"
```

### Creating an Agent

```python
from agent_runtime_core import (
    AgentRuntime,
    RunContext,
    RunResult,
    EventType,
    register_runtime,
)

class MyAgent(AgentRuntime):
    """A simple conversational agent."""

    @property
    def key(self) -> str:
        return "my-agent"

    async def run(self, ctx: RunContext) -> RunResult:
        # Access input messages
        messages = ctx.input_messages

        # Get an LLM client
        from agent_runtime_core.llm import get_llm_client
        llm = get_llm_client()

        # Generate a response
        response = await llm.generate(messages)

        # Emit events for observability
        await ctx.emit(EventType.ASSISTANT_MESSAGE, {
            "content": response.message["content"],
        })

        # Return the result
        return RunResult(
            final_output={"response": response.message["content"]},
            final_messages=[response.message],
        )

# Register the agent
register_runtime(MyAgent())
```

### Using Tools

```python
from agent_runtime_core import Tool, ToolRegistry, RunContext, RunResult

# Define tools
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny, 72°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

# Create a tool registry
tools = ToolRegistry()
tools.register(Tool.from_function(get_weather))
tools.register(Tool.from_function(search_web))

class ToolAgent(AgentRuntime):
    @property
    def key(self) -> str:
        return "tool-agent"

    async def run(self, ctx: RunContext) -> RunResult:
        from agent_runtime_core.llm import get_llm_client
        llm = get_llm_client()

        messages = list(ctx.input_messages)

        while True:
            # Generate with tools
            response = await llm.generate(
                messages,
                tools=tools.to_openai_format(),
            )

            messages.append(response.message)

            # Check for tool calls
            if not response.tool_calls:
                break

            # Execute tools
            for tool_call in response.tool_calls:
                result = await tools.execute(
                    tool_call["function"]["name"],
                    tool_call["function"]["arguments"],
                )

                await ctx.emit(EventType.TOOL_RESULT, {
                    "tool_call_id": tool_call["id"],
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result),
                })

        return RunResult(
            final_output={"response": response.message["content"]},
            final_messages=messages,
        )
```

### Running Agents

```python
from agent_runtime_core import AgentRunner, RunnerConfig, get_runtime
import asyncio

async def main():
    # Get a registered agent
    agent = get_runtime("my-agent")

    # Create a runner
    runner = AgentRunner(
        config=RunnerConfig(
            run_timeout_seconds=300,
            max_retries=3,
        )
    )

    # Execute a run
    result = await runner.execute(
        agent=agent,
        run_id="run-123",
        input_data={
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        },
    )

    print(result.final_output)

asyncio.run(main())
```

## Core Concepts

### AgentRuntime

The base class for all agents. Implement the `run` method to define your agent's behavior:

```python
class AgentRuntime(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        """Unique identifier for this agent."""
        pass

    @abstractmethod
    async def run(self, ctx: RunContext) -> RunResult:
        """Execute the agent logic."""
        pass
```

### RunContext

Provides access to the current run's state and utilities:

```python
class RunContext:
    run_id: UUID              # Unique run identifier
    input_messages: list      # Input messages
    metadata: dict            # Run metadata
    tools: ToolRegistry       # Available tools

    async def emit(self, event_type: EventType, payload: dict) -> None:
        """Emit an event."""

    async def checkpoint(self, state: dict) -> None:
        """Save a checkpoint."""

    def is_cancelled(self) -> bool:
        """Check if run was cancelled."""
```

### RunResult

The result of an agent run:

```python
@dataclass
class RunResult:
    final_output: dict           # Structured output
    final_messages: list = None  # Conversation history
    error: ErrorInfo = None      # Error details if failed
```

### Event Types

Built-in event types for observability:

- `EventType.RUN_STARTED` - Run execution began
- `EventType.RUN_SUCCEEDED` - Run completed successfully
- `EventType.RUN_FAILED` - Run failed with error
- `EventType.TOOL_CALL` - Tool was invoked
- `EventType.TOOL_RESULT` - Tool returned result
- `EventType.ASSISTANT_MESSAGE` - LLM generated message
- `EventType.CHECKPOINT` - State checkpoint saved

## Backend Options

### Queue Backends

```python
from agent_runtime_core.queue import InMemoryQueue, RedisQueue

# In-memory (for development)
queue = InMemoryQueue()

# Redis (for production)
queue = RedisQueue(redis_url="redis://localhost:6379/0")
```

### Event Bus Backends

```python
from agent_runtime_core.events import InMemoryEventBus, RedisEventBus

# In-memory
event_bus = InMemoryEventBus()

# Redis Pub/Sub
event_bus = RedisEventBus(redis_url="redis://localhost:6379/0")
```

### State Store Backends

```python
from agent_runtime_core.state import InMemoryStateStore, RedisStateStore, SQLiteStateStore

# In-memory
state = InMemoryStateStore()

# Redis
state = RedisStateStore(redis_url="redis://localhost:6379/0")

# SQLite (persistent, single-node)
state = SQLiteStateStore(db_path="./agent_state.db")
```

## Persistence

The persistence module provides storage for conversations, tasks, memory, and preferences with pluggable backends.

### File-Based Storage (Default)

```python
from agent_runtime_core.persistence import (
    PersistenceManager,
    PersistenceConfig,
    Scope,
)
from pathlib import Path

# Create manager with file-based storage
config = PersistenceConfig(project_dir=Path.cwd())
manager = PersistenceManager(config)

# Store memory (key-value)
await manager.memory.set("user_name", "Alice", scope=Scope.PROJECT)
name = await manager.memory.get("user_name")

# Store conversations
from agent_runtime_core.persistence import Conversation, Message
conv = Conversation(title="Chat 1")
conv.messages.append(Message(role="user", content="Hello!"))
await manager.conversations.save(conv)

# Store tasks
from agent_runtime_core.persistence import Task, TaskState
task = Task(name="Review code", conversation_id=conv.id)
await manager.tasks.save(task)
await manager.tasks.update(task.id, state=TaskState.COMPLETE)

# Store preferences
await manager.preferences.set("theme", "dark")
```

### Custom Backends (e.g., Django/Database)

The persistence layer is designed to be pluggable. Implement the abstract base classes for your backend:

```python
from agent_runtime_core.persistence import (
    MemoryStore,
    ConversationStore,
    TaskStore,
    PreferencesStore,
    PersistenceConfig,
    PersistenceManager,
)

class MyDatabaseMemoryStore(MemoryStore):
    def __init__(self, user):
        self.user = user

    async def get(self, key: str, scope=None) -> Optional[Any]:
        # Your database logic here
        pass

    async def set(self, key: str, value: Any, scope=None) -> None:
        # Your database logic here
        pass

    # ... implement other methods

# Three ways to configure custom backends:

# 1. Pre-instantiated stores (recommended for request-scoped)
config = PersistenceConfig(
    memory_store=MyDatabaseMemoryStore(user=request.user),
    conversation_store=MyDatabaseConversationStore(user=request.user),
)

# 2. Factory functions (for lazy instantiation)
config = PersistenceConfig(
    memory_store_factory=lambda: MyDatabaseMemoryStore(user=get_current_user()),
)

# 3. Classes with kwargs
config = PersistenceConfig(
    memory_store_class=MyDatabaseMemoryStore,
    memory_store_kwargs={"user": request.user},
)

manager = PersistenceManager(config)
```

### Persistence Data Models

```python
from agent_runtime_core.persistence import (
    # Conversation models
    Conversation,         # Chat conversation with messages
    ConversationMessage,  # Single message with branching support
    ToolCall,             # Tool invocation within a message
    ToolResult,           # Result of a tool call

    # Task models (with dependencies and checkpoints)
    Task,                 # Task with state, dependencies, checkpoints
    TaskList,             # Collection of tasks
    TaskState,            # NOT_STARTED, IN_PROGRESS, COMPLETE, CANCELLED

    # Knowledge models (optional)
    Fact,                 # Learned facts about user/project
    FactType,             # USER, PROJECT, PREFERENCE, CONTEXT, CUSTOM
    Summary,              # Conversation summaries
    Embedding,            # Vector embeddings for semantic search

    # Audit models (optional)
    AuditEntry,           # Interaction logs
    AuditEventType,       # CONVERSATION_START, TOOL_CALL, AGENT_ERROR, etc.
    ErrorRecord,          # Error history with resolution tracking
    ErrorSeverity,        # DEBUG, INFO, WARNING, ERROR, CRITICAL
    PerformanceMetric,    # Timing, token usage, etc.

    Scope,                # GLOBAL, PROJECT, SESSION
)
```

### Conversation Branching

Messages and conversations support branching for edit/regenerate workflows:

```python
from agent_runtime_core.persistence import Conversation, ConversationMessage
from uuid import uuid4

# Create a branched message (e.g., user edited their message)
branch_id = uuid4()
edited_msg = ConversationMessage(
    id=uuid4(),
    role="user",
    content="Updated question",
    parent_message_id=original_msg.id,  # Points to original
    branch_id=branch_id,
)

# Fork a conversation
forked_conv = Conversation(
    id=uuid4(),
    title="Forked conversation",
    parent_conversation_id=original_conv.id,
    active_branch_id=branch_id,
)
```

### Enhanced Tasks

Tasks support dependencies, checkpoints for resumable operations, and execution tracking:

```python
from agent_runtime_core.persistence import Task, TaskState
from uuid import uuid4
from datetime import datetime

task = Task(
    id=uuid4(),
    name="Process large dataset",
    description="Multi-step data processing",
    state=TaskState.IN_PROGRESS,

    # Dependencies - this task depends on others
    dependencies=[task1.id, task2.id],

    # Scheduling
    priority=10,  # Higher = more important
    due_at=datetime(2024, 12, 31),

    # Checkpoint for resumable operations
    checkpoint_data={"step": 5, "processed": 1000},
    checkpoint_at=datetime.utcnow(),

    # Execution tracking
    attempts=2,
    last_error="Temporary network failure",
)
```

### Optional: Knowledge Store

The KnowledgeStore is optional and must be explicitly configured. It stores facts, summaries, and embeddings:

```python
from agent_runtime_core.persistence import (
    KnowledgeStore, Fact, FactType, Summary, Embedding,
    PersistenceConfig, PersistenceManager,
)

# Implement your own KnowledgeStore
class MyKnowledgeStore(KnowledgeStore):
    async def save_fact(self, fact, scope=Scope.PROJECT):
        # Save to database
        ...

    async def get_fact(self, fact_id, scope=Scope.PROJECT):
        ...

    # ... implement other abstract methods

# Configure with optional store
config = PersistenceConfig(
    knowledge_store=MyKnowledgeStore(),
)
manager = PersistenceManager(config)

# Check if available before using
if manager.has_knowledge():
    await manager.knowledge.save_fact(Fact(
        id=uuid4(),
        key="user.preferred_language",
        value="Python",
        fact_type=FactType.PREFERENCE,
    ))
```

### Optional: Audit Store

The AuditStore is optional and tracks interaction logs, errors, and performance metrics:

```python
from agent_runtime_core.persistence import (
    AuditStore, AuditEntry, AuditEventType,
    ErrorRecord, ErrorSeverity, PerformanceMetric,
)

# Implement your own AuditStore
class MyAuditStore(AuditStore):
    async def log_event(self, entry, scope=Scope.PROJECT):
        # Log to database/file
        ...

    async def log_error(self, error, scope=Scope.PROJECT):
        ...

    async def record_metric(self, metric, scope=Scope.PROJECT):
        ...

    # ... implement other abstract methods

# Use in manager
config = PersistenceConfig(
    audit_store=MyAuditStore(),
)
manager = PersistenceManager(config)

if manager.has_audit():
    # Log an event
    await manager.audit.log_event(AuditEntry(
        id=uuid4(),
        event_type=AuditEventType.TOOL_CALL,
        action="Called search tool",
        details={"query": "python docs"},
    ))

    # Record performance metric
    await manager.audit.record_metric(PerformanceMetric(
        id=uuid4(),
        name="llm_latency",
        value=1250.5,
        unit="ms",
        tags={"model": "gpt-4"},
    ))
```

## LLM Clients

### OpenAI

```python
from agent_runtime_core.llm import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    default_model="gpt-4o",
)

response = await client.generate([
    {"role": "user", "content": "Hello!"}
])
```

### Anthropic

```python
from agent_runtime_core.llm import AnthropicClient

client = AnthropicClient(
    api_key="sk-ant-...",  # Or use ANTHROPIC_API_KEY env var
    default_model="claude-3-5-sonnet-20241022",
)
```

### LiteLLM (Any Provider)

```python
from agent_runtime_core.llm import LiteLLMClient

# Use any LiteLLM-supported model
client = LiteLLMClient(default_model="gpt-4o")
client = LiteLLMClient(default_model="claude-3-5-sonnet-20241022")
client = LiteLLMClient(default_model="ollama/llama2")
```

## Tracing & Observability

### Langfuse Integration

```python
from agent_runtime_core import configure

configure(
    langfuse_enabled=True,
    langfuse_public_key="pk-...",
    langfuse_secret_key="sk-...",
)
```

### Custom Trace Sink

```python
from agent_runtime_core import TraceSink

class MyTraceSink(TraceSink):
    async def trace(self, event: dict) -> None:
        # Send to your observability platform
        print(f"Trace: {event}")
```

## Integration with Django

For Django applications, use [django-agent-runtime](https://pypi.org/project/django-agent-runtime/) which provides:

- Django models for conversations, memory, tasks, and preferences
- Database-backed persistence stores
- REST API endpoints
- Server-Sent Events (SSE) for real-time streaming
- Management commands for running workers

```bash
pip install django-agent-runtime
```

## Testing

The library includes testing utilities for unit testing your agents:

```python
from agent_runtime_core.testing import (
    MockRunContext,
    MockLLMClient,
    create_test_context,
    run_agent_test,
)

# Create a mock context
ctx = create_test_context(
    input_messages=[{"role": "user", "content": "Hello!"}]
)

# Create a mock LLM client with predefined responses
mock_llm = MockLLMClient(responses=[
    {"role": "assistant", "content": "Hi there!"}
])

# Run your agent
result = await run_agent_test(MyAgent(), ctx)
assert result.final_output["response"] == "Hi there!"
```

## Step Executor

The `StepExecutor` provides a structured way to execute multi-step operations with automatic checkpointing, resume capability, retries, and progress reporting. Ideal for long-running agent tasks.

### Basic Usage

```python
from agent_runtime_core.steps import StepExecutor, Step

class MyAgent(AgentRuntime):
    async def run(self, ctx: RunContext) -> RunResult:
        executor = StepExecutor(ctx)

        results = await executor.run([
            Step("fetch", self.fetch_data),
            Step("process", self.process_data, retries=3),
            Step("validate", self.validate_results),
        ])

        return RunResult(final_output=results)

    async def fetch_data(self, ctx, state):
        # Fetch data from external API
        return {"items": [...]}

    async def process_data(self, ctx, state):
        # Access results from previous steps via state
        return {"processed": True}

    async def validate_results(self, ctx, state):
        return {"valid": True}
```

### Step Options

```python
Step(
    name="process",              # Unique step identifier
    fn=process_data,             # Async function(ctx, state) -> result
    retries=3,                   # Retry attempts on failure (default: 0)
    retry_delay=2.0,             # Seconds between retries (default: 1.0)
    timeout=30.0,                # Step timeout in seconds (optional)
    description="Process data",  # Human-readable description
    checkpoint=True,             # Save checkpoint after step (default: True)
)
```

### Resume from Checkpoint

Steps automatically checkpoint after completion. If execution is interrupted, it resumes from the last checkpoint:

```python
# First run - completes step1, fails during step2
executor = StepExecutor(ctx)
await executor.run([step1, step2, step3])  # Checkpoints after step1

# Second run - skips step1, resumes from step2
executor = StepExecutor(ctx)
await executor.run([step1, step2, step3])  # step1 skipped
```

### Custom State

Pass state between steps using `initial_state` and the `state` dict:

```python
async def step1(ctx, state):
    state["counter"] = 1
    return "done"

async def step2(ctx, state):
    state["counter"] += 1  # Access state from step1
    return state["counter"]

executor = StepExecutor(ctx)
results = await executor.run(
    [Step("step1", step1), Step("step2", step2)],
    initial_state={"counter": 0},
)
```

### Events

The executor emits events for observability:

- `EventType.STEP_STARTED` - Step execution began
- `EventType.STEP_COMPLETED` - Step completed successfully
- `EventType.STEP_FAILED` - Step failed after all retries
- `EventType.STEP_RETRYING` - Step is being retried
- `EventType.STEP_SKIPPED` - Step skipped (already completed)
- `EventType.PROGRESS_UPDATE` - Progress percentage update

## Multi-Agent Systems

The `multi_agent` module provides the "agent-as-tool" pattern, allowing agents to invoke other agents as tools. This enables router/dispatcher patterns, hierarchical agent systems, and specialist delegation.

### Core Concepts

**Invocation Modes:**
- `DELEGATE`: Sub-agent runs and returns result to parent (parent continues). Good for "get me an answer".
- `HANDOFF`: Control transfers completely to sub-agent (parent exits). Good for "transfer this customer to billing".

**Context Modes:**
- `FULL`: Complete conversation history passed to sub-agent (default)
- `SUMMARY`: Summarized context + current message (more efficient)
- `MESSAGE_ONLY`: Only the invocation message (clean isolation)

### Creating Agent Tools

Wrap any agent as a tool that can be called by other agents:

```python
from agent_runtime_core.multi_agent import (
    AgentTool,
    InvocationMode,
    ContextMode,
    invoke_agent,
    register_agent_tools,
)

# Define specialist agents
class BillingAgent(AgentRuntime):
    @property
    def key(self) -> str:
        return "billing-specialist"

    async def run(self, ctx: RunContext) -> RunResult:
        # Handle billing questions, refunds, payments
        ...

class TechSupportAgent(AgentRuntime):
    @property
    def key(self) -> str:
        return "tech-support"

    async def run(self, ctx: RunContext) -> RunResult:
        # Handle technical issues
        ...

# Wrap agents as tools
billing_tool = AgentTool(
    agent=BillingAgent(),
    name="billing_specialist",
    description="Handles billing questions, refunds, and payment issues. Use when customer has billing-related questions.",
    invocation_mode=InvocationMode.DELEGATE,
    context_mode=ContextMode.FULL,
)

tech_support_tool = AgentTool(
    agent=TechSupportAgent(),
    name="tech_support",
    description="Handles technical issues and troubleshooting. Use for technical problems.",
    invocation_mode=InvocationMode.HANDOFF,  # Transfer control completely
)
```

### Router Agent Pattern

Create a router agent that delegates to specialists:

```python
class RouterAgent(AgentRuntime):
    """Routes customer requests to appropriate specialist agents."""

    def __init__(self):
        self.agent_tools = [billing_tool, tech_support_tool]

    @property
    def key(self) -> str:
        return "customer-router"

    async def run(self, ctx: RunContext) -> RunResult:
        from agent_runtime_core.llm import get_llm_client
        llm = get_llm_client()

        # Create tool registry with agent-tools
        tools = ToolRegistry()
        messages = list(ctx.input_messages)

        # Register agent-tools as callable tools
        register_agent_tools(
            registry=tools,
            agent_tools=self.agent_tools,
            get_conversation_history=lambda: messages,
            parent_ctx=ctx,
        )

        # Add system prompt for routing
        system_message = {
            "role": "system",
            "content": """You are a customer service router. Analyze the customer's
            request and delegate to the appropriate specialist:
            - billing_specialist: For billing, payments, refunds
            - tech_support: For technical issues (this will transfer the customer)

            If you can answer directly, do so. Otherwise, use the appropriate tool."""
        }

        while True:
            response = await llm.generate(
                [system_message] + messages,
                tools=tools.to_openai_format(),
            )

            messages.append(response.message)

            if not response.tool_calls:
                # No tool call - respond directly
                await ctx.emit(EventType.ASSISTANT_MESSAGE, {
                    "content": response.content,
                })
                break

            # Execute tool calls (which may invoke sub-agents)
            for tool_call in response.tool_calls:
                result = await tools.execute_with_events(tool_call, ctx)

                # Check for handoff
                if isinstance(result, dict) and result.get("handoff"):
                    # Sub-agent took over - we're done
                    return RunResult(
                        final_output=result.get("final_output", {}),
                        final_messages=messages,
                    )

                # Add result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.get("response", result)),
                })

        return RunResult(
            final_output={"response": response.content},
            final_messages=messages,
        )
```

### Direct Invocation

You can also invoke agents directly without the tool pattern:

```python
from agent_runtime_core.multi_agent import invoke_agent

# Invoke a sub-agent directly
result = await invoke_agent(
    agent_tool=billing_tool,
    message="Customer wants a refund for order #123",
    parent_ctx=ctx,
    conversation_history=messages,
    additional_context="Customer has been waiting 2 weeks",
)

if result.handoff:
    # Sub-agent took over
    return RunResult(final_output=result.run_result.final_output)
else:
    # Use the response
    print(result.response)
```

### Events

Multi-agent invocations emit events for observability:

- `sub_agent.start` - Sub-agent invocation started
- `sub_agent.end` - Sub-agent invocation completed
- `tool.call` with `is_agent_tool: true` - Agent-tool was called
- `tool.result` with `is_agent_tool: true` - Agent-tool returned

### AgentTool Options

```python
AgentTool(
    agent=my_agent,                          # Required: The agent to wrap
    name="specialist",                       # Required: Tool name
    description="When to use this agent",    # Required: Shown to parent LLM
    invocation_mode=InvocationMode.DELEGATE, # DELEGATE or HANDOFF
    context_mode=ContextMode.FULL,           # FULL, SUMMARY, or MESSAGE_ONLY
    max_turns=10,                            # Optional: Limit sub-agent turns
    input_schema={...},                      # Optional: Custom input schema
    metadata={"category": "billing"},        # Optional: Additional metadata
)
```

## API Reference

### Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_provider` | str | `"openai"` | LLM provider: openai, anthropic, litellm |
| `default_model` | str | `"gpt-4o"` | Default model to use |
| `queue_backend` | str | `"memory"` | Queue backend: memory, redis |
| `event_bus_backend` | str | `"memory"` | Event bus: memory, redis |
| `state_store_backend` | str | `"memory"` | State store: memory, redis, sqlite |
| `redis_url` | str | `None` | Redis connection URL |
| `langfuse_enabled` | bool | `False` | Enable Langfuse tracing |

### Registry Functions

```python
register_runtime(runtime: AgentRuntime) -> None
get_runtime(key: str) -> AgentRuntime
list_runtimes() -> list[str]
unregister_runtime(key: str) -> None
clear_registry() -> None
```

### Persistence Functions

```python
from agent_runtime_core.persistence import (
    configure_persistence,
    get_persistence_manager,
)

# Configure global persistence
configure_persistence(
    memory_store_class=MyMemoryStore,
    project_dir=Path.cwd(),
)

# Get the global manager
manager = get_persistence_manager()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
