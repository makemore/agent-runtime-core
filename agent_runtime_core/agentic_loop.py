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
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Awaitable, Union

from agent_runtime_core.interfaces import (
    RunContext,
    EventType,
    LLMClient,
    LLMResponse,
)
from agent_runtime_core.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Cost Estimation Configuration
# =============================================================================

# Pricing per 1M tokens (input/output) - updated Jan 2026
# These are approximate and should be updated as pricing changes
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Default fallback
    "default": {"input": 3.00, "output": 15.00},
}


def _get_model_pricing(model: str) -> dict:
    """Get pricing for a model, with fallback to default."""
    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Try prefix match (e.g., "gpt-4o-2024-08-06" -> "gpt-4o")
    for key in MODEL_PRICING:
        if model.startswith(key):
            return MODEL_PRICING[key]
    return MODEL_PRICING["default"]


def _estimate_cost(usage: dict, model: str) -> float:
    """Estimate cost in USD from usage dict and model."""
    pricing = _get_model_pricing(model)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def _format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.3f}"


@dataclass
class UsageStats:
    """Accumulated usage statistics for the agentic loop."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0

    def add_llm_call(self, usage: dict, model: str):
        """Add usage from an LLM call."""
        self.llm_calls += 1
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_cost += _estimate_cost(usage, model)

    def add_tool_call(self):
        """Record a tool call."""
        self.tool_calls += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }


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

    usage_stats: Optional[UsageStats] = None
    """Accumulated usage statistics across all LLM calls (if debug mode enabled)."""


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
    ensure_final_response: bool = False,
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
        ensure_final_response: If True, ensures a summary is generated when tools
            were used but the final response is empty or very short. This is useful
            for agents that should always provide a summary of what was accomplished.
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
            ensure_final_response=True,  # Guarantees a summary
        )
    """
    iteration = 0
    final_content = ""
    last_response: Optional[LLMResponse] = None
    consecutive_errors = 0
    max_consecutive_errors = 3  # Bail out if tool keeps failing

    # Initialize usage tracking (enabled in debug mode)
    debug_mode = get_config().debug
    usage_stats = UsageStats() if debug_mode else None
    effective_model = model or "unknown"

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

        # Track usage in debug mode
        if debug_mode and usage_stats:
            # Get model from response if available, otherwise use effective_model
            resp_model = response.model or effective_model
            usage_stats.add_llm_call(response.usage, resp_model)

            # Print debug info
            prompt_tokens = response.usage.get("prompt_tokens", 0)
            completion_tokens = response.usage.get("completion_tokens", 0)
            call_cost = _estimate_cost(response.usage, resp_model)

            print(f"[agentic-loop] 💰 LLM Call #{usage_stats.llm_calls}:", flush=True)
            print(f"[agentic-loop]    Model: {resp_model}", flush=True)
            print(f"[agentic-loop]    Tokens: {prompt_tokens:,} in / {completion_tokens:,} out", flush=True)
            print(f"[agentic-loop]    Cost: {_format_cost(call_cost)}", flush=True)
            print(f"[agentic-loop]    Running total: {usage_stats.total_prompt_tokens:,} in / {usage_stats.total_completion_tokens:,} out = {_format_cost(usage_stats.total_cost)}", flush=True)
        
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
                
                # Track tool call in debug mode
                if debug_mode and usage_stats:
                    usage_stats.add_tool_call()
                    print(f"[agentic-loop] 🔧 Tool #{usage_stats.tool_calls}: {tool_name}", flush=True)

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

                    # Generate a summary if ensure_final_response is enabled
                    if ensure_final_response:
                        logger.info("Generating summary after error exit because ensure_final_response=True")
                        print("[agentic-loop] Generating summary after error exit", flush=True)
                        summary = await _generate_task_summary(llm, messages, model, **llm_kwargs)
                        if summary:
                            final_content = f"{summary}\n\n---\n\n⚠️ Note: The task ended early due to repeated errors. Last error: {error_msg}"
                        else:
                            final_content = f"I encountered repeated errors while trying to complete this task. The last error was: {error_msg}"
                    else:
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

                    # Print final summary in debug mode
                    if debug_mode and usage_stats:
                        print(f"[agentic-loop] ═══════════════════════════════════════════", flush=True)
                        print(f"[agentic-loop] 📊 FINAL USAGE SUMMARY (error exit)", flush=True)
                        print(f"[agentic-loop]    LLM calls: {usage_stats.llm_calls}", flush=True)
                        print(f"[agentic-loop]    Tool calls: {usage_stats.tool_calls}", flush=True)
                        print(f"[agentic-loop]    Total tokens: {usage_stats.total_prompt_tokens:,} in / {usage_stats.total_completion_tokens:,} out", flush=True)
                        print(f"[agentic-loop]    Estimated cost: {_format_cost(usage_stats.total_cost)}", flush=True)
                        print(f"[agentic-loop] ═══════════════════════════════════════════", flush=True)

                    return AgenticLoopResult(
                        final_content=final_content,
                        messages=messages,
                        iterations=iteration,
                        usage=last_response.usage if last_response else {},
                        usage_stats=usage_stats,
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

    # Check if we need to ensure a final response (summary)
    if ensure_final_response:
        # Check if tools were used during this run
        tools_were_used = any(
            msg.get("role") == "assistant" and msg.get("tool_calls")
            for msg in messages
        )

        # If tools were used but final response is empty or very short, generate a summary
        if tools_were_used and (not final_content or len(final_content.strip()) < 50):
            logger.info("Generating summary because tools were used but final response was empty/short")
            print("[agentic-loop] Generating summary - tools were used but no final response", flush=True)

            summary = await _generate_task_summary(llm, messages, model, **llm_kwargs)
            if summary:
                final_content = summary
                # Emit the summary as an assistant message
                if emit_events:
                    await ctx.emit(EventType.ASSISTANT_MESSAGE, {
                        "content": summary,
                        "role": "assistant",
                    })
                # Add to messages
                messages.append({"role": "assistant", "content": summary})

    # Print final summary in debug mode
    if debug_mode and usage_stats:
        print(f"[agentic-loop] ═══════════════════════════════════════════", flush=True)
        print(f"[agentic-loop] 📊 FINAL USAGE SUMMARY", flush=True)
        print(f"[agentic-loop]    Iterations: {iteration}", flush=True)
        print(f"[agentic-loop]    LLM calls: {usage_stats.llm_calls}", flush=True)
        print(f"[agentic-loop]    Tool calls: {usage_stats.tool_calls}", flush=True)
        print(f"[agentic-loop]    Total tokens: {usage_stats.total_prompt_tokens:,} in / {usage_stats.total_completion_tokens:,} out", flush=True)
        print(f"[agentic-loop]    Estimated cost: {_format_cost(usage_stats.total_cost)}", flush=True)
        print(f"[agentic-loop] ═══════════════════════════════════════════", flush=True)

    return AgenticLoopResult(
        final_content=final_content,
        messages=messages,
        iterations=iteration,
        usage=last_response.usage if last_response else {},
        usage_stats=usage_stats,
    )


async def _generate_task_summary(
    llm: LLMClient,
    messages: list[dict],
    model: Optional[str] = None,
    **llm_kwargs,
) -> str:
    """
    Generate a summary of what was accomplished based on the conversation history.

    This is called when ensure_final_response=True and tools were used but
    no meaningful final response was provided.

    Args:
        llm: The LLM client to use
        messages: The conversation history including tool calls and results
        model: Model to use
        **llm_kwargs: Additional kwargs for the LLM

    Returns:
        A summary string of what was accomplished
    """
    # Build a summary of tool calls and their results
    tool_summary_parts = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    name = tc.get("function", {}).get("name", "unknown")
                else:
                    name = getattr(tc, "name", "unknown")
                tool_summary_parts.append(f"- Called: {name}")
        elif msg.get("role") == "tool":
            content = msg.get("content", "")
            # Truncate long results
            if len(content) > 200:
                content = content[:200] + "..."
            tool_summary_parts.append(f"  Result: {content}")

    tool_summary = "\n".join(tool_summary_parts[-20:])  # Last 20 entries to avoid token limits

    summary_prompt = f"""Based on the conversation above, provide a brief summary of what was accomplished.

Here's a summary of the tools that were called:
{tool_summary}

Please provide a clear, concise summary (2-4 sentences) of:
1. What actions were taken
2. What was accomplished or changed
3. Any important results or next steps

Start your response directly with the summary - do not include phrases like "Here's a summary" or "Based on the conversation"."""

    # Create a simplified message list for the summary request
    summary_messages = [
        {"role": "system", "content": "You are a helpful assistant that provides clear, concise summaries of completed tasks."},
        {"role": "user", "content": summary_prompt},
    ]

    try:
        response = await llm.generate(
            summary_messages,
            model=model,
            **llm_kwargs,
        )
        return response.message.get("content", "")
    except Exception as e:
        logger.exception("Failed to generate task summary")
        return f"Task completed. (Summary generation failed: {e})"
