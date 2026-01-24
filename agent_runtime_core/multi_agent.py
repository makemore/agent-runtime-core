"""
Multi-agent support for agent_runtime_core.

This module provides the "agent-as-tool" pattern, allowing agents to invoke
other agents as tools. This enables:
- Router/dispatcher patterns
- Hierarchical agent systems
- Specialist delegation

Two invocation modes are supported:
- DELEGATE: Sub-agent runs and returns result to parent (parent continues)
- HANDOFF: Control transfers completely to sub-agent (parent exits)

Context passing is configurable:
- FULL: Complete conversation history passed to sub-agent (default)
- SUMMARY: Summarized context + current message
- MESSAGE_ONLY: Only the invocation message

Example:
    from agent_runtime_core.multi_agent import (
        AgentTool,
        InvocationMode,
        ContextMode,
        invoke_agent,
    )
    
    # Define a sub-agent as a tool
    billing_agent_tool = AgentTool(
        agent=billing_agent,
        name="billing_specialist",
        description="Handles billing questions, refunds, and payment issues",
        invocation_mode=InvocationMode.DELEGATE,
        context_mode=ContextMode.FULL,
    )
    
    # Invoke it
    result = await invoke_agent(
        agent_tool=billing_agent_tool,
        message="Customer wants a refund for order #123",
        parent_ctx=ctx,
        conversation_history=messages,
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING
from uuid import UUID, uuid4

from agent_runtime_core.interfaces import (
    AgentRuntime,
    Message,
    RunContext,
    RunResult,
    Tool,
    ToolRegistry,
    EventType,
)

if TYPE_CHECKING:
    from agent_runtime_core.contexts import InMemoryRunContext

logger = logging.getLogger(__name__)


class InvocationMode(str, Enum):
    """
    How the sub-agent is invoked.
    
    DELEGATE: Sub-agent runs, returns result to parent. Parent continues
              its execution with the result. Good for "get me an answer".
              
    HANDOFF: Control transfers completely to sub-agent. Parent's run ends
             and sub-agent takes over the conversation. Good for "transfer
             this customer to billing".
    """
    DELEGATE = "delegate"
    HANDOFF = "handoff"


class ContextMode(str, Enum):
    """
    What context is passed to the sub-agent.
    
    FULL: Complete conversation history. Sub-agent sees everything the
          parent has seen. Best for sensitive contexts where nothing
          should be forgotten. This is the default.
          
    SUMMARY: A summary of the conversation + the current message.
             More efficient but may lose nuance.
             
    MESSAGE_ONLY: Only the invocation message. Clean isolation but
                  sub-agent lacks context.
    """
    FULL = "full"
    SUMMARY = "summary"
    MESSAGE_ONLY = "message_only"


@dataclass
class AgentTool:
    """
    Wraps an agent to be used as a tool by another agent.
    
    This is the core abstraction for multi-agent systems. Any agent can
    be wrapped as a tool and added to another agent's tool registry.
    
    Attributes:
        agent: The agent runtime to invoke
        name: Tool name (how the parent agent calls it)
        description: When to use this agent (shown to parent LLM)
        invocation_mode: DELEGATE or HANDOFF
        context_mode: How much context to pass (FULL, SUMMARY, MESSAGE_ONLY)
        max_turns: Optional limit on sub-agent turns (for DELEGATE mode)
        input_schema: Optional custom input schema (defaults to message + context)
        metadata: Additional metadata for the tool
    """
    agent: AgentRuntime
    name: str
    description: str
    invocation_mode: InvocationMode = InvocationMode.DELEGATE
    context_mode: ContextMode = ContextMode.FULL
    max_turns: Optional[int] = None
    input_schema: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    
    def to_tool_schema(self) -> dict:
        """
        Generate the OpenAI-format tool schema for this agent-tool.
        
        The schema allows the parent agent to invoke this sub-agent
        with a message and optional context override.
        """
        if self.input_schema:
            parameters = self.input_schema
        else:
            # Default schema: message + optional context
            parameters = {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message or task to send to this agent",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context to include",
                    },
                },
                "required": ["message"],
            }
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


@dataclass
class AgentInvocationResult:
    """
    Result from invoking a sub-agent.
    
    Attributes:
        response: The sub-agent's final response text
        messages: All messages from the sub-agent's run
        handoff: True if this was a handoff (parent should exit)
        run_result: The full RunResult from the sub-agent
        sub_agent_key: The key of the agent that was invoked
    """
    response: str
    messages: list[Message]
    handoff: bool
    run_result: RunResult
    sub_agent_key: str


class SubAgentContext:
    """
    RunContext implementation for sub-agent invocations.
    
    Wraps the parent context but with modified input messages
    based on the context mode.
    """
    
    def __init__(
        self,
        parent_ctx: RunContext,
        input_messages: list[Message],
        sub_run_id: Optional[UUID] = None,
    ):
        self._parent_ctx = parent_ctx
        self._input_messages = input_messages
        self._run_id = sub_run_id or uuid4()
        self._state: Optional[dict] = None
    
    @property
    def run_id(self) -> UUID:
        return self._run_id
    
    @property
    def conversation_id(self) -> Optional[UUID]:
        # Sub-agent shares the same conversation
        return self._parent_ctx.conversation_id
    
    @property
    def input_messages(self) -> list[Message]:
        return self._input_messages
    
    @property
    def params(self) -> dict:
        return self._parent_ctx.params
    
    @property
    def metadata(self) -> dict:
        # Add sub-agent metadata
        meta = dict(self._parent_ctx.metadata)
        meta["parent_run_id"] = str(self._parent_ctx.run_id)
        meta["is_sub_agent"] = True
        return meta
    
    @property
    def tool_registry(self) -> ToolRegistry:
        # Sub-agent uses its own tools, not parent's
        # This is set by the agent itself
        return ToolRegistry()
    
    async def emit(self, event_type: EventType | str, payload: dict) -> None:
        """Emit events through parent context with sub-agent tagging."""
        # Tag the event as coming from a sub-agent
        tagged_payload = dict(payload)
        tagged_payload["sub_agent_run_id"] = str(self._run_id)
        tagged_payload["parent_run_id"] = str(self._parent_ctx.run_id)
        await self._parent_ctx.emit(event_type, tagged_payload)
    
    async def emit_user_message(self, content: str) -> None:
        """Emit a user-visible message."""
        await self.emit(EventType.ASSISTANT_MESSAGE, {
            "content": content,
            "role": "assistant",
        })
    
    async def emit_error(self, error: str, details: dict = None) -> None:
        """Emit an error event."""
        await self.emit(EventType.ERROR, {
            "error": error,
            "details": details or {},
        })
    
    async def checkpoint(self, state: dict) -> None:
        """Save state checkpoint."""
        self._state = state
        # Could also delegate to parent for persistence
    
    async def get_state(self) -> Optional[dict]:
        """Get last checkpointed state."""
        return self._state
    
    def cancelled(self) -> bool:
        """Check if parent has been cancelled."""
        return self._parent_ctx.cancelled()


def build_sub_agent_messages(
    context_mode: ContextMode,
    message: str,
    conversation_history: list[Message],
    additional_context: Optional[str] = None,
) -> list[Message]:
    """
    Build the input messages for a sub-agent based on context mode.
    
    Args:
        context_mode: How much context to include
        message: The invocation message from the parent
        conversation_history: Full conversation history from parent
        additional_context: Optional extra context string
        
    Returns:
        List of messages to pass to the sub-agent
    """
    if context_mode == ContextMode.FULL:
        # Include full history + new message
        messages = list(conversation_history)
        
        # Build the user message
        user_content = message
        if additional_context:
            user_content = f"{additional_context}\n\n{message}"
        
        messages.append({
            "role": "user",
            "content": user_content,
        })
        return messages
    
    elif context_mode == ContextMode.SUMMARY:
        # TODO: Implement summarization
        # For now, fall back to including last few messages
        recent_messages = conversation_history[-5:] if conversation_history else []
        
        user_content = message
        if additional_context:
            user_content = f"{additional_context}\n\n{message}"
        
        return list(recent_messages) + [{
            "role": "user",
            "content": user_content,
        }]
    
    else:  # MESSAGE_ONLY
        user_content = message
        if additional_context:
            user_content = f"{additional_context}\n\n{message}"
        
        return [{
            "role": "user",
            "content": user_content,
        }]


async def invoke_agent(
    agent_tool: AgentTool,
    message: str,
    parent_ctx: RunContext,
    conversation_history: Optional[list[Message]] = None,
    additional_context: Optional[str] = None,
) -> AgentInvocationResult:
    """
    Invoke a sub-agent as a tool.
    
    This is the main entry point for agent-to-agent invocation.
    
    Args:
        agent_tool: The AgentTool wrapping the sub-agent
        message: The message/task to send to the sub-agent
        parent_ctx: The parent agent's run context
        conversation_history: Full conversation history (for FULL context mode)
        additional_context: Optional extra context to include
        
    Returns:
        AgentInvocationResult with the sub-agent's response
        
    Example:
        result = await invoke_agent(
            agent_tool=billing_specialist,
            message="Process refund for order #123",
            parent_ctx=ctx,
            conversation_history=messages,
        )
        
        if result.handoff:
            # Sub-agent took over, return its result
            return RunResult(
                final_output=result.run_result.final_output,
                final_messages=result.messages,
            )
        else:
            # Got a response, continue parent execution
            print(f"Billing says: {result.response}")
    """
    logger.info(
        f"Invoking sub-agent '{agent_tool.name}' "
        f"(mode={agent_tool.invocation_mode.value}, "
        f"context={agent_tool.context_mode.value})"
    )
    
    # Build messages based on context mode
    history = conversation_history or []
    sub_messages = build_sub_agent_messages(
        context_mode=agent_tool.context_mode,
        message=message,
        conversation_history=history,
        additional_context=additional_context,
    )
    
    # Create sub-agent context
    sub_ctx = SubAgentContext(
        parent_ctx=parent_ctx,
        input_messages=sub_messages,
    )
    
    # Emit event for sub-agent invocation (tool call format)
    await parent_ctx.emit(EventType.TOOL_CALL, {
        "name": agent_tool.name,
        "arguments": {"message": message, "context": additional_context},
        "is_agent_tool": True,
        "sub_agent_key": agent_tool.agent.key,
        "invocation_mode": agent_tool.invocation_mode.value,
    })

    # Also emit a custom sub_agent.start event for UI display
    # Get agent name from the agent if available
    agent_name = getattr(agent_tool.agent, 'name', None) or agent_tool.name
    await parent_ctx.emit("sub_agent.start", {
        "sub_agent_key": agent_tool.agent.key,
        "agent_name": agent_name,
        "tool_name": agent_tool.name,
        "invocation_mode": agent_tool.invocation_mode.value,
        "context_mode": agent_tool.context_mode.value,
    })

    # Run the sub-agent
    try:
        run_result = await agent_tool.agent.run(sub_ctx)
    except Exception as e:
        logger.exception(f"Sub-agent '{agent_tool.name}' failed")
        # Emit error event
        await parent_ctx.emit(EventType.TOOL_RESULT, {
            "name": agent_tool.name,
            "is_agent_tool": True,
            "error": str(e),
        })
        # Emit sub_agent.end with error
        await parent_ctx.emit("sub_agent.end", {
            "sub_agent_key": agent_tool.agent.key,
            "agent_name": agent_name,
            "tool_name": agent_tool.name,
            "error": str(e),
        })
        raise

    # Extract response
    response = run_result.final_output.get("response", "")
    if not response and run_result.final_messages:
        # Try to get from last assistant message
        for msg in reversed(run_result.final_messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
                break

    # Emit result event (tool result format)
    await parent_ctx.emit(EventType.TOOL_RESULT, {
        "name": agent_tool.name,
        "is_agent_tool": True,
        "sub_agent_key": agent_tool.agent.key,
        "response": response[:500] if response else "",  # Truncate for event
        "handoff": agent_tool.invocation_mode == InvocationMode.HANDOFF,
    })

    # Also emit a custom sub_agent.end event for UI display
    await parent_ctx.emit("sub_agent.end", {
        "sub_agent_key": agent_tool.agent.key,
        "agent_name": agent_name,
        "tool_name": agent_tool.name,
        "success": True,
        "handoff": agent_tool.invocation_mode == InvocationMode.HANDOFF,
    })
    
    return AgentInvocationResult(
        response=response,
        messages=run_result.final_messages,
        handoff=agent_tool.invocation_mode == InvocationMode.HANDOFF,
        run_result=run_result,
        sub_agent_key=agent_tool.agent.key,
    )


def create_agent_tool_handler(
    agent_tool: AgentTool,
    get_conversation_history: Callable[[], list[Message]],
    parent_ctx: RunContext,
) -> Callable:
    """
    Create a tool handler function for an AgentTool.
    
    This creates a handler that can be registered in a ToolRegistry,
    allowing the agent-tool to be called like any other tool.
    
    Args:
        agent_tool: The AgentTool to create a handler for
        get_conversation_history: Function that returns current conversation
        parent_ctx: The parent agent's context
        
    Returns:
        Async handler function compatible with ToolRegistry
        
    Example:
        handler = create_agent_tool_handler(
            billing_agent_tool,
            lambda: current_messages,
            ctx,
        )
        
        registry.register(Tool(
            name=billing_agent_tool.name,
            description=billing_agent_tool.description,
            parameters=billing_agent_tool.to_tool_schema()["function"]["parameters"],
            handler=handler,
        ))
    """
    async def handler(message: str, context: Optional[str] = None) -> dict:
        result = await invoke_agent(
            agent_tool=agent_tool,
            message=message,
            parent_ctx=parent_ctx,
            conversation_history=get_conversation_history(),
            additional_context=context,
        )
        
        if result.handoff:
            # Signal to the parent that this is a handoff
            return {
                "handoff": True,
                "response": result.response,
                "sub_agent": result.sub_agent_key,
                "final_output": result.run_result.final_output,
            }
        else:
            return {
                "response": result.response,
                "sub_agent": result.sub_agent_key,
            }
    
    return handler


def register_agent_tools(
    registry: ToolRegistry,
    agent_tools: list[AgentTool],
    get_conversation_history: Callable[[], list[Message]],
    parent_ctx: RunContext,
) -> None:
    """
    Register multiple agent-tools in a ToolRegistry.
    
    Convenience function to add several sub-agents as tools.
    
    Args:
        registry: The ToolRegistry to add tools to
        agent_tools: List of AgentTools to register
        get_conversation_history: Function returning current conversation
        parent_ctx: Parent agent's context
        
    Example:
        register_agent_tools(
            registry=self.tools,
            agent_tools=[billing_agent_tool, support_agent_tool],
            get_conversation_history=lambda: self._messages,
            parent_ctx=ctx,
        )
    """
    for agent_tool in agent_tools:
        handler = create_agent_tool_handler(
            agent_tool=agent_tool,
            get_conversation_history=get_conversation_history,
            parent_ctx=parent_ctx,
        )
        
        schema = agent_tool.to_tool_schema()
        
        registry.register(Tool(
            name=agent_tool.name,
            description=agent_tool.description,
            parameters=schema["function"]["parameters"],
            handler=handler,
            metadata={
                "is_agent_tool": True,
                "sub_agent_key": agent_tool.agent.key,
                "invocation_mode": agent_tool.invocation_mode.value,
                "context_mode": agent_tool.context_mode.value,
            },
        ))

