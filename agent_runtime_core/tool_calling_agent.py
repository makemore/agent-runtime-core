"""
ToolCallingAgent - A base class for agents that use tool calling.

This eliminates the boilerplate of implementing the tool-calling loop
in every agent. Just define your system prompt and tools, and the base
class handles the rest.
"""

import json
import logging
from abc import abstractmethod
from typing import Optional

from agent_runtime_core.interfaces import (
    AgentRuntime,
    RunContext,
    RunResult,
    EventType,
    ToolRegistry,
    LLMClient,
)

logger = logging.getLogger(__name__)


class ToolCallingAgent(AgentRuntime):
    """
    Base class for agents that use tool calling.
    
    Handles the standard tool-calling loop so you don't have to implement it
    in every agent. Just override the abstract properties and you're done.
    
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
        """
        return 10
    
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
    
    async def on_tool_result(self, ctx: RunContext, tool_name: str, result: any) -> any:
        """
        Hook called after each tool execution.
        
        Override to transform or validate tool results.
        Can return a modified result.
        """
        return result
    
    async def run(self, ctx: RunContext) -> RunResult:
        """
        Execute the agent with tool calling support.
        
        This implements the standard tool-calling loop:
        1. Build messages with system prompt
        2. Call LLM with tools
        3. If tool calls, execute them and loop
        4. If no tool calls, return final response
        """
        logger.info(f"[{self.key}] Starting run, input messages: {len(ctx.input_messages)}")
        
        # Call before_run hook
        await self.before_run(ctx)
        
        # Get LLM client
        llm = self.get_llm_client()
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + ctx.input_messages
        
        logger.info(f"[{self.key}] Built {len(messages)} messages (including system prompt)")
        
        # Run the agent loop (tool calling)
        iteration = 0
        final_response = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[{self.key}] Iteration {iteration}/{self.max_iterations}")
            
            # Generate response with tools
            logger.info(f"[{self.key}] Calling LLM...")
            response = await llm.generate(
                messages=messages,
                tools=self.tools.to_openai_format(),
                model=self.model,
                temperature=self.temperature,
            )
            logger.info(f"[{self.key}] LLM response received, tool_calls: {bool(response.message.get('tool_calls'))}")
            
            # Check if the model wants to call tools
            if response.message.get('tool_calls'):
                # Add the assistant message with tool calls
                messages.append(response.message)
                
                # Execute the tools
                tool_results = []
                for tool_call in response.message.get('tool_calls'):
                    # Emit tool call event
                    await ctx.emit(EventType.TOOL_CALL, {
                        "tool_name": tool_call["function"]["name"],
                        "tool_args": json.loads(tool_call["function"]["arguments"]),
                        "tool_call_id": tool_call["id"],
                    })
                    
                    # Call before_tool_call hook
                    await self.on_tool_call(ctx, tool_call["function"]["name"], json.loads(tool_call["function"]["arguments"]))
                    
                    # Execute the tool
                    result = await self.tools.execute(
                        tool_call["function"]["name"],
                        json.loads(tool_call["function"]["arguments"]),
                    )
                    
                    # Call after_tool_result hook
                    result = await self.on_tool_result(ctx, tool_call["function"]["name"], result)
                    
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "result": result,
                    })
                    
                    # Emit tool result event
                    await ctx.emit(EventType.TOOL_RESULT, {
                        "tool_name": tool_call["function"]["name"],
                        "tool_call_id": tool_call["id"],
                        "result": result,
                    })
                
                # Add tool results to messages
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": str(tr["result"]),
                    })
            else:
                # No tool calls - we have the final response
                final_response = response.message["content"]
                logger.info(f"[{self.key}] Final response received: {final_response[:100] if final_response else 'None'}...")
                break
        
        # Emit the final assistant message
        if final_response:
            logger.info(f"[{self.key}] Emitting ASSISTANT_MESSAGE event")
            await ctx.emit(EventType.ASSISTANT_MESSAGE, {
                "content": final_response,
            })
            logger.info(f"[{self.key}] Event emitted successfully")
        else:
            logger.warning(f"[{self.key}] No final response to emit!")
        
        logger.info(f"[{self.key}] Returning RunResult")
        result = RunResult(
            final_output={"response": final_response},
            final_messages=messages,
            usage=response.usage if response else {},
        )
        
        # Call after_run hook
        result = await self.after_run(ctx, result)
        
        return result
