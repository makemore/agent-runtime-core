"""
Testing utilities for agent runtimes.

This module provides tools for testing agent implementations:
- MockRunContext: A concrete RunContext for unit tests
- MockLLMClient: A mock LLM client with predefined responses
- AgentTestCase: Base test class with common helpers
- LLMEvaluator: Use LLM to evaluate agent responses

Example usage:
    from agent_runtime_core.testing import MockRunContext, MockLLMClient, AgentTestCase
    
    class TestMyAgent(AgentTestCase):
        async def test_agent_responds(self):
            ctx = self.create_context("Hello, agent!")
            result = await self.agent.run(ctx)
            self.assertIn("response", result.final_output)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, AsyncIterator
from uuid import UUID, uuid4
import json

from .interfaces import (
    AgentRuntime,
    EventType,
    LLMClient,
    LLMResponse,
    LLMStreamChunk,
    Message,
    RunContext,
    RunResult,
    Tool,
    ToolRegistry,
)


@dataclass
class MockRunContext:
    """
    A concrete implementation of RunContext for testing.
    
    Use this in unit tests to provide a context to your agent
    without needing the full runtime infrastructure.
    
    Example:
        ctx = MockRunContext(
            input_messages=[{"role": "user", "content": "Hello"}],
            metadata={"user_id": "123"}
        )
        result = await my_agent.run(ctx)
    """
    
    input_messages: list[Message] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    run_id: UUID = field(default_factory=uuid4)
    conversation_id: Optional[UUID] = None
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    
    # Internal state
    _events: list[tuple[str, dict]] = field(default_factory=list)
    _checkpoints: list[dict] = field(default_factory=list)
    _cancelled: bool = False
    
    async def emit(self, event_type: EventType | str, payload: dict) -> None:
        """Record emitted events for later inspection."""
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        self._events.append((event_name, payload))
    
    async def checkpoint(self, state: dict) -> None:
        """Save a checkpoint."""
        self._checkpoints.append(state)
    
    async def get_state(self) -> Optional[dict]:
        """Get the last checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None
    
    def cancelled(self) -> bool:
        """Check if cancelled."""
        return self._cancelled
    
    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
    
    # Test helpers
    def get_events(self, event_type: Optional[str] = None) -> list[tuple[str, dict]]:
        """Get recorded events, optionally filtered by type."""
        if event_type is None:
            return self._events
        return [(t, p) for t, p in self._events if t == event_type]
    
    def get_checkpoints(self) -> list[dict]:
        """Get all checkpoints."""
        return self._checkpoints
    
    def clear(self) -> None:
        """Clear recorded events and checkpoints."""
        self._events.clear()
        self._checkpoints.clear()
        self._cancelled = False


@dataclass
class MockLLMResponse:
    """A predefined response for MockLLMClient."""
    content: str
    tool_calls: Optional[list[dict]] = None
    finish_reason: str = "stop"


class MockLLMClient(LLMClient):
    """
    A mock LLM client for testing.
    
    Configure with predefined responses or a response function.
    
    Example:
        # Simple predefined responses
        client = MockLLMClient(responses=[
            MockLLMResponse(content="Hello!"),
            MockLLMResponse(content="How can I help?"),
        ])
        
        # Dynamic responses based on input
        def respond(messages):
            if "weather" in messages[-1]["content"].lower():
                return MockLLMResponse(content="It's sunny!")
            return MockLLMResponse(content="I don't know.")
        
        client = MockLLMClient(response_fn=respond)
    """
    
    def __init__(
        self,
        responses: Optional[list[MockLLMResponse]] = None,
        response_fn: Optional[Callable[[list[Message]], MockLLMResponse]] = None,
        default_response: str = "Mock response",
    ):
        self._responses = responses or []
        self._response_fn = response_fn
        self._default_response = default_response
        self._call_count = 0
        self._calls: list[dict] = []
    
    async def generate(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock response."""
        # Record the call
        self._calls.append({
            "messages": messages,
            "model": model,
            "tools": tools,
            "kwargs": kwargs,
        })
        
        # Get response
        if self._response_fn:
            mock_resp = self._response_fn(messages)
        elif self._call_count < len(self._responses):
            mock_resp = self._responses[self._call_count]
        else:
            mock_resp = MockLLMResponse(content=self._default_response)
        
        self._call_count += 1
        
        # Build message
        message: Message = {
            "role": "assistant",
            "content": mock_resp.content,
        }
        if mock_resp.tool_calls:
            message["tool_calls"] = mock_resp.tool_calls
        
        return LLMResponse(
            message=message,
            model=model or "mock-model",
            finish_reason=mock_resp.finish_reason,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
    
    async def stream(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream a mock response (yields content in chunks)."""
        response = await self.generate(messages, model=model, tools=tools, **kwargs)
        content = response.message.get("content", "")
        
        # Yield content in chunks
        for i in range(0, len(content), 10):
            yield LLMStreamChunk(delta=content[i:i+10])
        
        yield LLMStreamChunk(finish_reason="stop", usage=response.usage)
    
    # Test helpers
    def get_calls(self) -> list[dict]:
        """Get all recorded calls."""
        return self._calls
    
    def get_call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count
    
    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self._calls.clear()


class LLMEvaluator:
    """
    Use an LLM to evaluate agent responses.
    
    This is useful for testing that agent responses meet certain criteria
    without having to write brittle string matching tests.
    
    Example:
        evaluator = LLMEvaluator(openai_client)
        
        passed, explanation = await evaluator.evaluate(
            user_query="What's the weather?",
            agent_response="It's currently 72°F and sunny in San Francisco.",
            criteria="The response should include temperature and weather conditions"
        )
        
        assert passed, f"Evaluation failed: {explanation}"
    """
    
    def __init__(self, llm_client: LLMClient, model: str = "gpt-4o-mini"):
        self._client = llm_client
        self._model = model
    
    async def evaluate(
        self,
        user_query: str,
        agent_response: str,
        criteria: str,
    ) -> tuple[bool, str]:
        """
        Evaluate an agent response against criteria.
        
        Args:
            user_query: The original user query
            agent_response: The agent's response
            criteria: What the response should satisfy
            
        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        eval_prompt = f"""You are evaluating an AI assistant's response.

User Query: {user_query}

Agent Response: {agent_response}

Evaluation Criteria: {criteria}

Does the response meet the criteria? Answer with just "PASS" or "FAIL" followed by a brief explanation."""

        response = await self._client.generate(
            messages=[{"role": "user", "content": eval_prompt}],
            model=self._model,
            temperature=0,
        )
        
        result = response.message.get("content", "FAIL Unknown error")
        passed = result.strip().upper().startswith("PASS")
        return passed, result
    
    async def evaluate_tool_usage(
        self,
        user_query: str,
        tool_calls: list[dict],
        expected_tools: list[str],
    ) -> tuple[bool, str]:
        """
        Evaluate whether the agent used the expected tools.
        
        Args:
            user_query: The original user query
            tool_calls: List of tool calls made by the agent
            expected_tools: List of tool names that should have been called
            
        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        tool_names = [tc.get("function", {}).get("name", tc.get("name", "unknown")) 
                      for tc in tool_calls]
        
        missing = set(expected_tools) - set(tool_names)
        if missing:
            return False, f"Missing expected tools: {missing}. Called: {tool_names}"
        
        return True, f"All expected tools were called: {tool_names}"


def create_test_context(
    message: str,
    *,
    tools: Optional[list[Tool]] = None,
    metadata: Optional[dict] = None,
    params: Optional[dict] = None,
) -> MockRunContext:
    """
    Convenience function to create a test context.
    
    Example:
        ctx = create_test_context("Hello, agent!", tools=[my_tool])
        result = await agent.run(ctx)
    """
    registry = ToolRegistry()
    if tools:
        for tool in tools:
            registry.register(tool)
    
    return MockRunContext(
        input_messages=[{"role": "user", "content": message}],
        tool_registry=registry,
        metadata=metadata or {},
        params=params or {},
    )


async def run_agent_test(
    agent: AgentRuntime,
    message: str,
    *,
    tools: Optional[list[Tool]] = None,
    metadata: Optional[dict] = None,
) -> tuple[RunResult, MockRunContext]:
    """
    Run an agent with a test message and return both result and context.
    
    Example:
        result, ctx = await run_agent_test(my_agent, "Hello!")
        assert "greeting" in result.final_output
        assert len(ctx.get_events()) > 0
    """
    ctx = create_test_context(message, tools=tools, metadata=metadata)
    result = await agent.run(ctx)
    return result, ctx
