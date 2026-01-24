"""
JsonAgentRuntime - A runtime that loads agent configuration from JSON.

This allows running agents defined in the portable AgentConfig format,
either from a JSON file or from a Django revision snapshot.

Example:
    # Load from file
    config = AgentConfig.from_file("my_agent.json")
    runtime = JsonAgentRuntime(config)

    # Or load directly
    runtime = JsonAgentRuntime.from_file("my_agent.json")

    # Run the agent
    result = await runtime.run(ctx)

Multi-Agent Support:
    The runtime supports sub-agent tools defined in the config. Sub-agents
    can be embedded (agent_config) or referenced by slug (agent_slug).

    # Config with embedded sub-agent
    config = AgentConfig(
        slug="triage-agent",
        sub_agent_tools=[
            SubAgentToolConfig(
                name="billing_specialist",
                description="Handle billing questions",
                agent_config=AgentConfig(slug="billing-agent", ...),
            )
        ]
    )
"""

import importlib
import logging
from typing import Any, Callable, Optional

from agent_runtime_core.interfaces import (
    AgentRuntime,
    RunContext,
    RunResult,
    Tool,
    ToolDefinition,
)
from agent_runtime_core.agentic_loop import run_agentic_loop
from agent_runtime_core.config_schema import AgentConfig, ToolConfig, SubAgentToolConfig

logger = logging.getLogger(__name__)


def resolve_function(function_path: str) -> Callable:
    """
    Resolve a function path like 'myapp.services.orders.lookup_order' to the actual callable.
    
    Args:
        function_path: Dotted path to the function
        
    Returns:
        The callable function
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function doesn't exist in the module
    """
    parts = function_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid function path: {function_path}. Expected 'module.function' format.")
    
    module_path, function_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


class ConfiguredTool(Tool):
    """A tool created from ToolConfig that resolves function_path at runtime."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self._function: Optional[Callable] = None
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.config.name,
            description=self.config.description,
            parameters=self.config.parameters,
        )
    
    def _get_function(self) -> Callable:
        """Lazily resolve the function."""
        if self._function is None:
            self._function = resolve_function(self.config.function_path)
        return self._function
    
    async def execute(self, args: dict, ctx: RunContext) -> Any:
        """Execute the tool by calling the resolved function."""
        func = self._get_function()

        # Check if function is async
        if hasattr(func, "__call__"):
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**args)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(**args))

        return func(**args)


class SubAgentTool(Tool):
    """
    A tool that delegates to another agent (sub-agent).

    This implements the "agent-as-tool" pattern where one agent can
    invoke another agent as if it were a tool.
    """

    def __init__(
        self,
        config: SubAgentToolConfig,
        agent_registry: Optional[dict[str, "JsonAgentRuntime"]] = None,
    ):
        """
        Initialize a sub-agent tool.

        Args:
            config: The sub-agent tool configuration
            agent_registry: Optional registry of pre-loaded agent runtimes
                           (for embedded configs or external lookup)
        """
        self.config = config
        self.agent_registry = agent_registry or {}
        self._runtime: Optional["JsonAgentRuntime"] = None

    @property
    def definition(self) -> ToolDefinition:
        """Get the tool definition for this sub-agent."""
        # Sub-agent tools take a single 'message' parameter
        # Note: The handler here is a placeholder - actual execution goes through
        # execute() which receives the RunContext. The runtime handles this specially.
        async def _placeholder_handler(**kwargs):
            raise RuntimeError("SubAgentTool.execute() should be called directly with context")

        return ToolDefinition(
            name=self.config.name,
            description=self.config.description,
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message or task to send to the sub-agent",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context for the sub-agent",
                    },
                },
                "required": ["message"],
            },
            handler=_placeholder_handler,
        )

    def _get_runtime(self) -> "JsonAgentRuntime":
        """Get or create the sub-agent runtime."""
        if self._runtime is not None:
            return self._runtime

        # Try embedded config first
        if self.config.agent_config:
            self._runtime = JsonAgentRuntime(self.config.agent_config)
            return self._runtime

        # Try registry lookup by slug
        if self.config.agent_slug:
            if self.config.agent_slug in self.agent_registry:
                self._runtime = self.agent_registry[self.config.agent_slug]
                return self._runtime
            raise ValueError(
                f"Sub-agent '{self.config.agent_slug}' not found in registry. "
                f"Available: {list(self.agent_registry.keys())}"
            )

        raise ValueError(
            f"Sub-agent tool '{self.config.name}' has no agent_config or agent_slug"
        )

    async def execute(self, args: dict, ctx: RunContext) -> Any:
        """
        Execute the sub-agent with the given message.

        Args:
            args: Tool arguments (message, optional context)
            ctx: The parent run context

        Returns:
            The sub-agent's response
        """
        message = args.get("message", "")
        additional_context = args.get("context", "")

        runtime = self._get_runtime()

        # Build messages for sub-agent based on context_mode
        sub_messages = self._build_sub_agent_messages(message, additional_context, ctx)

        # Create a sub-context for the sub-agent
        # Note: In a full implementation, you'd want to track parent-child relationships
        from agent_runtime_core.interfaces import RunContext as RC
        sub_ctx = RC(
            run_id=f"{ctx.run_id}-sub-{self.config.name}",
            conversation_id=ctx.conversation_id,
            input_messages=sub_messages,
            params=ctx.params,
        )

        # Run the sub-agent
        logger.info(f"Invoking sub-agent '{self.config.name}' ({runtime.key})")
        result = await runtime.run(sub_ctx)

        # Extract the response
        response = result.final_output.get("response", "")
        if not response and result.final_messages:
            # Try to get from last assistant message
            for msg in reversed(result.final_messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    response = msg["content"]
                    break

        return response

    def _build_sub_agent_messages(
        self,
        message: str,
        additional_context: str,
        ctx: RunContext,
    ) -> list[dict]:
        """
        Build the message list for the sub-agent based on context_mode.

        Args:
            message: The message to send
            additional_context: Optional additional context
            ctx: Parent run context

        Returns:
            List of messages for the sub-agent
        """
        context_mode = self.config.context_mode

        if context_mode == "full":
            # Pass full conversation history
            messages = list(ctx.input_messages)
            # Add the delegation message
            if additional_context:
                messages.append({
                    "role": "user",
                    "content": f"{message}\n\nAdditional context: {additional_context}",
                })
            else:
                messages.append({"role": "user", "content": message})
            return messages

        elif context_mode == "summary":
            # Summarize context (simplified - in production you'd use LLM)
            summary = f"Previous conversation context: {len(ctx.input_messages)} messages exchanged."
            return [
                {"role": "system", "content": summary},
                {"role": "user", "content": message},
            ]

        else:  # "message_only" or default
            # Just the message, no context
            content = message
            if additional_context:
                content = f"{message}\n\nContext: {additional_context}"
            return [{"role": "user", "content": content}]


class JsonAgentRuntime(AgentRuntime):
    """
    An agent runtime that loads its configuration from AgentConfig.

    This provides a portable way to define and run agents using JSON configuration,
    without requiring Django or any specific framework.

    Supports multi-agent systems through sub_agent_tools, which allow this agent
    to delegate to other agents as if they were tools.
    """

    def __init__(
        self,
        config: AgentConfig,
        agent_registry: Optional[dict[str, "JsonAgentRuntime"]] = None,
    ):
        """
        Initialize the runtime with an AgentConfig.

        Args:
            config: The agent configuration
            agent_registry: Optional registry of pre-loaded agent runtimes
                           for sub-agent lookup by slug
        """
        self.config = config
        self.agent_registry = agent_registry or {}
        self._tools: list[Tool] = []
        self._tools_loaded = False

        # Build registry from embedded sub-agent configs
        self._build_embedded_agent_registry()

    def _build_embedded_agent_registry(self) -> None:
        """Build registry entries for embedded sub-agent configs."""
        for sub_tool in self.config.sub_agent_tools:
            if sub_tool.agent_config:
                slug = sub_tool.agent_config.slug
                if slug not in self.agent_registry:
                    # Create runtime for embedded config (recursively)
                    self.agent_registry[slug] = JsonAgentRuntime(
                        sub_tool.agent_config,
                        agent_registry=self.agent_registry,
                    )

    @property
    def key(self) -> str:
        return self.config.slug

    @classmethod
    def from_file(cls, path: str) -> "JsonAgentRuntime":
        """Load a runtime from a JSON file."""
        config = AgentConfig.from_file(path)
        return cls(config)

    @classmethod
    def from_json(cls, json_str: str) -> "JsonAgentRuntime":
        """Load a runtime from a JSON string."""
        config = AgentConfig.from_json(json_str)
        return cls(config)

    @classmethod
    def from_dict(cls, data: dict) -> "JsonAgentRuntime":
        """Load a runtime from a dictionary."""
        config = AgentConfig.from_dict(data)
        return cls(config)

    @classmethod
    def from_system_export(cls, data: dict) -> "JsonAgentRuntime":
        """
        Load a runtime from an exported multi-agent system.

        This handles the system export format which has entry_agent at the top level.

        Args:
            data: Exported system config (from AgentSystemVersion.export_config)

        Returns:
            JsonAgentRuntime for the entry agent with all sub-agents wired up
        """
        entry_agent_data = data.get("entry_agent")
        if not entry_agent_data:
            raise ValueError("System export must have 'entry_agent' key")

        config = AgentConfig.from_dict(entry_agent_data)
        return cls(config)

    def _load_tools(self) -> list[Tool]:
        """Load and resolve all tools from config, including sub-agent tools."""
        if self._tools_loaded:
            return self._tools

        self._tools = []

        # Load regular function tools
        for tool_config in self.config.tools:
            try:
                tool = ConfiguredTool(tool_config)
                # Validate that the function can be resolved
                tool._get_function()
                self._tools.append(tool)
                logger.debug(f"Loaded tool: {tool_config.name}")
            except Exception as e:
                logger.error(f"Failed to load tool {tool_config.name}: {e}")
                raise

        # Load sub-agent tools
        for sub_tool_config in self.config.sub_agent_tools:
            try:
                tool = SubAgentTool(sub_tool_config, agent_registry=self.agent_registry)
                self._tools.append(tool)
                logger.debug(f"Loaded sub-agent tool: {sub_tool_config.name}")
            except Exception as e:
                logger.error(f"Failed to load sub-agent tool {sub_tool_config.name}: {e}")
                raise

        self._tools_loaded = True
        return self._tools

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including knowledge."""
        parts = []

        # Add base system prompt
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)

        # Add always-included knowledge
        for knowledge in self.config.knowledge:
            if knowledge.inclusion_mode == "always" and knowledge.content:
                parts.append(f"\n\n## {knowledge.name}\n{knowledge.content}")

        return "\n".join(parts)

    async def run(self, ctx: RunContext) -> RunResult:
        """
        Run the agent using the agentic loop.

        Args:
            ctx: The run context with conversation history and state

        Returns:
            RunResult with the agent's response
        """
        from agent_runtime_core.llm import get_llm_client

        # Load tools
        tools = self._load_tools()

        # Build system prompt with knowledge
        system_prompt = self._build_system_prompt()

        # Get model settings
        model = self.config.model
        model_settings = self.config.model_settings or {}

        # Get LLM client
        llm = get_llm_client(model=model)

        # Build messages list with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(ctx.input_messages)

        # Convert tools to OpenAI format
        tool_schemas = None
        tool_map = {}
        if tools:
            tool_schemas = []
            for tool in tools:
                tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.definition.name,
                        "description": tool.definition.description,
                        "parameters": tool.definition.parameters,
                    }
                })
                tool_map[tool.definition.name] = tool

        # Create tool executor
        async def execute_tool(name: str, args: dict) -> Any:
            if name not in tool_map:
                raise ValueError(f"Unknown tool: {name}")
            return await tool_map[name].execute(args, ctx)

        # Run the agentic loop
        result = await run_agentic_loop(
            llm=llm,
            messages=messages,
            tools=tool_schemas,
            execute_tool=execute_tool,
            ctx=ctx,
            model=model,
            **model_settings,
        )

        return RunResult(
            final_output={"response": result.final_content},
            final_messages=result.messages,
            usage=result.usage,
        )

    def get_tools(self) -> list[Tool]:
        """Get the list of configured tools (including sub-agent tools)."""
        return self._load_tools()

    def get_system_prompt(self) -> str:
        """Get the full system prompt including knowledge."""
        return self._build_system_prompt()

    def get_sub_agent_tools(self) -> list[SubAgentTool]:
        """Get only the sub-agent tools."""
        return [t for t in self._load_tools() if isinstance(t, SubAgentTool)]

    def get_sub_agent_runtimes(self) -> dict[str, "JsonAgentRuntime"]:
        """
        Get all sub-agent runtimes available to this agent.

        Returns:
            Dict mapping agent slug to JsonAgentRuntime
        """
        return dict(self.agent_registry)

    def has_sub_agents(self) -> bool:
        """Check if this agent has any sub-agent tools."""
        return len(self.config.sub_agent_tools) > 0
