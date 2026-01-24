"""
Reusable agentic loop for tool-calling agents.

This module provides a flexible `run_agentic_loop` function that handles
the standard tool-calling pattern:
1. Call LLM with tools
2. If tool calls, execute them and loop back
3. If no tool calls, return final response

This can be used by any agent implementation without requiring inheritance.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Awaitable, Union

from agent_runtime_core.interfaces import (
    RunContext,
    EventType,
    LLMClient,
    LLMResponse,
)

logger = logging.getLogger(__name__)

# Type alias for tool executor function
ToolExecutor = Callable[[str, dict], Awaitable[Any]]


@dataclass
class AgenticLoopResult:
    """Result from running the agentic loop."""
    
    final_content: str
    """The final text response from the LLM."""
    
    messages: list[dict]
    """All messages including tool calls and results."""
    
    iterations: int
    """Number of iterations the loop ran."""
    
    usage: dict
    """Token usage from the final LLM call."""


async def run_agentic_loop(
    llm: LLMClient,
    messages: list[dict],
    tools: Optional[list[dict]],
    execute_tool: ToolExecutor,
    ctx: RunContext,
    *,
    model: Optional[str] = None,
    max_iterations: int = 15,
    emit_events: bool = True,
    **llm_kwargs,
) -> AgenticLoopResult:
    """
    Run the standard agentic tool-calling loop.
    
    This handles the common pattern of:
    1. Call LLM with available tools
    2. If LLM returns tool calls, execute them
    3. Add tool results to messages and loop back to step 1
    4. If LLM returns a text response (no tool calls), return it
    
    Args:
        llm: The LLM client to use for generation
        messages: Initial messages (should include system prompt)
        tools: List of tool schemas in OpenAI format, or None for no tools
        execute_tool: Async function that executes a tool: (name, args) -> result
        ctx: Run context for emitting events
        model: Model to use (passed to LLM client)
        max_iterations: Maximum loop iterations to prevent infinite loops
        emit_events: Whether to emit TOOL_CALL and TOOL_RESULT events
        **llm_kwargs: Additional kwargs passed to llm.generate()
    
    Returns:
        AgenticLoopResult with final content, messages, and metadata
    
    Example:
        async def my_tool_executor(name: str, args: dict) -> Any:
            if name == "get_weather":
                return {"temp": 72, "conditions": "sunny"}
            raise ValueError(f"Unknown tool: {name}")
        
        result = await run_agentic_loop(
            llm=my_llm_client,
            messages=[{"role": "system", "content": "You are helpful."}],
            tools=[{"type": "function", "function": {...}}],
            execute_tool=my_tool_executor,
            ctx=ctx,
            model="gpt-4o",
        )
    """
    iteration = 0
    final_content = ""
    last_response: Optional[LLMResponse] = None
    consecutive_errors = 0
    max_consecutive_errors = 3  # Bail out if tool keeps failing

    while iteration < max_iterations:
        iteration += 1
        print(f"[agentic-loop] Iteration {iteration}/{max_iterations}, messages={len(messages)}", flush=True)
        logger.debug(f"Agentic loop iteration {iteration}/{max_iterations}")

        # Call LLM
        if tools:
            response = await llm.generate(
                messages,
                model=model,
                tools=tools,
                **llm_kwargs,
            )
        else:
            response = await llm.generate(
                messages,
                model=model,
                **llm_kwargs,
            )
        
        last_response = response
        
        # Check for tool calls
        if response.tool_calls:
            # Add assistant message with tool calls to conversation
            messages.append(response.message)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                # Debug: log raw tool call to help diagnose empty args issue
                print(f"[agentic-loop] Raw tool_call type={type(tool_call).__name__}", flush=True)
                if hasattr(tool_call, '_data'):
                    print(f"[agentic-loop] tool_call._data={tool_call._data}", flush=True)

                # Handle both ToolCall objects (with .id, .name, .arguments) and dicts
                if hasattr(tool_call, 'id') and not isinstance(tool_call, dict):
                    # ToolCall object
                    tool_call_id = tool_call.id
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments
                    print(f"[agentic-loop] Parsed: name={tool_name}, args={tool_args}", flush=True)
                else:
                    # Dict format
                    tool_call_id = tool_call.get("id")
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                    logger.debug(f"Dict tool_call: id={tool_call_id}, name={tool_name}, args_str={tool_args_str}")
                    # Parse arguments (handle both string and dict)
                    if isinstance(tool_args_str, dict):
                        tool_args = tool_args_str
                    else:
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool args: {tool_args_str}")
                            tool_args = {}
                
                # Emit tool call event
                if emit_events:
                    await ctx.emit(EventType.TOOL_CALL, {
                        "id": tool_call_id,
                        "name": tool_name,
                        "arguments": tool_args,
                    })
                
                # Execute the tool
                try:
                    result = await execute_tool(tool_name, tool_args)
                    # Reset error counter on success
                    if not isinstance(result, dict) or "error" not in result:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                except Exception as e:
                    logger.exception(f"Error executing tool {tool_name}")
                    result = {"error": str(e)}
                    consecutive_errors += 1

                # Emit tool result event
                if emit_events:
                    await ctx.emit(EventType.TOOL_RESULT, {
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "result": result,
                    })

                # Add tool result to messages
                result_str = json.dumps(result) if not isinstance(result, str) else result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                })

                # Check for too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                    logger.warning(f"Aborting agentic loop after {consecutive_errors} consecutive tool errors: {error_msg}")

                    # Emit error event for run history
                    if emit_events:
                        await ctx.emit(EventType.ERROR, {
                            "error": f"Tool loop aborted after {consecutive_errors} consecutive errors",
                            "last_error": error_msg,
                            "tool_name": tool_name,
                            "iterations": iteration,
                        })

                    # Add error to messages for conversation history
                    final_content = f"I encountered repeated errors while trying to complete this task. The last error was: {error_msg}"
                    messages.append({
                        "role": "assistant",
                        "content": final_content,
                    })

                    if emit_events:
                        await ctx.emit(EventType.ASSISTANT_MESSAGE, {
                            "content": final_content,
                            "role": "assistant",
                        })

                    return AgenticLoopResult(
                        final_content=final_content,
                        messages=messages,
                        iterations=iteration,
                        usage=last_response.usage if last_response else {},
                    )

            # Continue the loop to get next response
            continue
        
        # No tool calls - we have the final response
        final_content = response.message.get("content", "")
        messages.append(response.message)

        # Emit assistant message event for the final response
        if emit_events and final_content:
            await ctx.emit(EventType.ASSISTANT_MESSAGE, {
                "content": final_content,
                "role": "assistant",
            })

        break
    
    return AgenticLoopResult(
        final_content=final_content,
        messages=messages,
        iterations=iteration,
        usage=last_response.usage if last_response else {},
    )

