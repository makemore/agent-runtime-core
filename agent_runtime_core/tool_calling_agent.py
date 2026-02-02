"""
ToolCallingAgent - A base class for agents that use tool calling.

This eliminates the boilerplate of implementing the tool-calling loop
in every agent. Just define your system prompt and tools, and the base
class handles the rest.

Uses run_agentic_loop internally for the actual loop logic.
"""

import logging
from abc import abstractmethod
from typing import Any, Optional

from agent_runtime_core.interfaces import (
    AgentRuntime,
    RunContext,
    RunResult,
    EventType,
    ToolRegistry,
    LLMClient,
)
from agent_runtime_core.agentic_loop import run_agentic_loop

logger = logging.getLogger(__name__)


class ToolCallingAgent(AgentRuntime):
    """
    Base class for agents that use tool calling.

    Handles the standard tool-calling loop so you don't have to implement it
    in every agent. Just override the abstract properties and you're done.

    Uses run_agentic_loop internally, with hooks for customization.

    Example:
        class MyAgent(ToolCallingAgent):
            @property
            def key(self) -> str:
                return "my-agent"

            @property
            def system_prompt(self) -> str:
                return "You are a helpful assistant..."

            @property
            def tools(self) -> ToolRegistry:
                return create_my_tools()
    """

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        System prompt for the agent.

        This is prepended to the conversation messages.
        """
        ...

    @property
    @abstractmethod
    def tools(self) -> ToolRegistry:
        """
        Tools available to the agent.

        Return a ToolRegistry with all tools registered.
        """
        ...

    @property
    def max_iterations(self) -> int:
        """
        Maximum number of tool-calling iterations.

        Override to change the default limit.
        Default uses the value from config (default: 50).
        """
        from agent_runtime_core.config import get_config
        return get_config().max_iterations

    @property
    def model(self) -> Optional[str]:
        """
        Model to use for this agent.

        If None, uses the default model from configuration.
        Override to use a specific model.
        """
        return None

    @property
    def temperature(self) -> Optional[float]:
        """
        Temperature for LLM generation.

        If None, uses the LLM client's default.
        Override to set a specific temperature.
        """
        return None

    def get_llm_client(self) -> LLMClient:
        """
        Get the LLM client to use.

        Override to customize LLM client selection.
        Default uses the configured client.
        """
        from agent_runtime_core.llm import get_llm_client
        return get_llm_client()

    async def before_run(self, ctx: RunContext) -> None:
        """
        Hook called before the agent run starts.

        Override to add custom initialization logic.
        """
        pass

    async def after_run(self, ctx: RunContext, result: RunResult) -> RunResult:
        """
        Hook called after the agent run completes.

        Override to add custom finalization logic.
        Can modify the result before returning.
        """
        return result

    async def on_tool_call(self, ctx: RunContext, tool_name: str, tool_args: dict) -> None:
        """
        Hook called before each tool execution.

        Override to add custom logic (logging, validation, etc.).
        """
        pass

    async def on_tool_result(self, ctx: RunContext, tool_name: str, result: Any) -> Any:
        """
        Hook called after each tool execution.

        Override to transform or validate tool results.
        Can return a modified result.
        """
        return result

    async def run(self, ctx: RunContext) -> RunResult:
        """
        Execute the agent with tool calling support.

        Uses run_agentic_loop internally with hooks for customization.
        """
        logger.debug(f"[{self.key}] Starting run, input messages: {len(ctx.input_messages)}")

        # Call before_run hook
        await self.before_run(ctx)

        # Get LLM client
        llm = self.get_llm_client()

        # Build messages with system prompt
        # Use system_prompt_with_memory if available (from MemoryEnabledAgent mixin)
        prompt = (
            self.system_prompt_with_memory
            if hasattr(self, 'system_prompt_with_memory')
            else self.system_prompt
        )
        messages = [
            {"role": "system", "content": prompt}
        ] + list(ctx.input_messages)

        # Create tool executor that calls our hooks
        async def execute_tool(name: str, args: dict) -> Any:
            # Call before hook
            await self.on_tool_call(ctx, name, args)

            # Execute the tool
            result = await self.tools.execute(name, args)

            # Call after hook (can transform result)
            result = await self.on_tool_result(ctx, name, result)

            return result

        # Get tool schemas
        tool_schemas = self.tools.to_openai_format() if self.tools.list_tools() else None

        # Build LLM kwargs
        llm_kwargs = {}
        if self.temperature is not None:
            llm_kwargs["temperature"] = self.temperature

        # Run the agentic loop
        # Note: agentic_loop emits ASSISTANT_MESSAGE for the final response
        loop_result = await run_agentic_loop(
            llm=llm,
            messages=messages,
            tools=tool_schemas,
            execute_tool=execute_tool,
            ctx=ctx,
            model=self.model,
            max_iterations=self.max_iterations,
            emit_events=True,
            **llm_kwargs,
        )

        result = RunResult(
            final_output={"response": loop_result.final_content},
            final_messages=loop_result.messages,
            usage=loop_result.usage,
        )

        # Call after_run hook
        result = await self.after_run(ctx, result)

        return result
