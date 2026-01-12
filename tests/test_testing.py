"""Tests for the testing utilities module."""

import pytest
from uuid import UUID

from agent_runtime_core import (
    AgentRuntime,
    EventType,
    RunContext,
    RunResult,
    Tool,
    ToolRegistry,
)
from agent_runtime_core.testing import (
    MockRunContext,
    MockLLMClient,
    MockLLMResponse,
    create_test_context,
    run_agent_test,
)


class TestMockRunContext:
    """Tests for MockRunContext."""
    
    def test_default_values(self):
        """Test that MockRunContext has sensible defaults."""
        ctx = MockRunContext()
        
        assert isinstance(ctx.run_id, UUID)
        assert ctx.conversation_id is None
        assert ctx.input_messages == []
        assert ctx.params == {}
        assert ctx.metadata == {}
        assert isinstance(ctx.tool_registry, ToolRegistry)
        assert ctx.cancelled() is False
    
    def test_custom_values(self):
        """Test MockRunContext with custom values."""
        ctx = MockRunContext(
            input_messages=[{"role": "user", "content": "Hello"}],
            params={"temperature": 0.7},
            metadata={"user_id": "123"},
        )
        
        assert len(ctx.input_messages) == 1
        assert ctx.input_messages[0]["content"] == "Hello"
        assert ctx.params["temperature"] == 0.7
        assert ctx.metadata["user_id"] == "123"
    
    @pytest.mark.asyncio
    async def test_emit_events(self):
        """Test event emission and retrieval."""
        ctx = MockRunContext()
        
        await ctx.emit(EventType.RUN_STARTED, {"agent": "test"})
        await ctx.emit(EventType.TOOL_CALL, {"tool": "search"})
        await ctx.emit(EventType.RUN_SUCCEEDED, {"result": "done"})
        
        # Get all events
        events = ctx.get_events()
        assert len(events) == 3
        
        # Filter by type
        tool_events = ctx.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0][1]["tool"] == "search"
    
    @pytest.mark.asyncio
    async def test_checkpoints(self):
        """Test checkpoint save and retrieval."""
        ctx = MockRunContext()
        
        # No checkpoint initially
        state = await ctx.get_state()
        assert state is None
        
        # Save checkpoints
        await ctx.checkpoint({"step": 1})
        await ctx.checkpoint({"step": 2})
        
        # Get latest
        state = await ctx.get_state()
        assert state["step"] == 2
        
        # Get all
        checkpoints = ctx.get_checkpoints()
        assert len(checkpoints) == 2
    
    def test_cancellation(self):
        """Test cancellation flag."""
        ctx = MockRunContext()
        
        assert ctx.cancelled() is False
        ctx.cancel()
        assert ctx.cancelled() is True
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing recorded data."""
        ctx = MockRunContext()
        
        await ctx.emit(EventType.RUN_STARTED, {})
        await ctx.checkpoint({"step": 1})
        ctx.cancel()
        
        ctx.clear()
        
        assert ctx.get_events() == []
        assert ctx.get_checkpoints() == []
        assert ctx.cancelled() is False


class TestMockLLMClient:
    """Tests for MockLLMClient."""
    
    @pytest.mark.asyncio
    async def test_default_response(self):
        """Test default response when no responses configured."""
        client = MockLLMClient()
        
        response = await client.generate([{"role": "user", "content": "Hi"}])
        
        assert response.message["role"] == "assistant"
        assert response.message["content"] == "Mock response"
        assert response.model == "mock-model"
    
    @pytest.mark.asyncio
    async def test_predefined_responses(self):
        """Test cycling through predefined responses."""
        client = MockLLMClient(responses=[
            MockLLMResponse(content="First"),
            MockLLMResponse(content="Second"),
        ])
        
        r1 = await client.generate([{"role": "user", "content": "1"}])
        r2 = await client.generate([{"role": "user", "content": "2"}])
        r3 = await client.generate([{"role": "user", "content": "3"}])
        
        assert r1.message["content"] == "First"
        assert r2.message["content"] == "Second"
        assert r3.message["content"] == "Mock response"  # Falls back to default
    
    @pytest.mark.asyncio
    async def test_response_function(self):
        """Test dynamic response function."""
        def respond(messages):
            content = messages[-1].get("content", "")
            if "weather" in content.lower():
                return MockLLMResponse(content="It's sunny!")
            return MockLLMResponse(content="I don't know.")
        
        client = MockLLMClient(response_fn=respond)
        
        r1 = await client.generate([{"role": "user", "content": "What's the weather?"}])
        r2 = await client.generate([{"role": "user", "content": "Hello"}])
        
        assert r1.message["content"] == "It's sunny!"
        assert r2.message["content"] == "I don't know."
    
    @pytest.mark.asyncio
    async def test_tool_calls(self):
        """Test responses with tool calls."""
        client = MockLLMClient(responses=[
            MockLLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'}
                }]
            )
        ])
        
        response = await client.generate([{"role": "user", "content": "Search"}])
        
        assert response.message["tool_calls"] is not None
        assert len(response.message["tool_calls"]) == 1
        assert response.message["tool_calls"][0]["function"]["name"] == "search"
    
    @pytest.mark.asyncio
    async def test_call_tracking(self):
        """Test that calls are recorded."""
        client = MockLLMClient()
        
        await client.generate(
            [{"role": "user", "content": "Hi"}],
            model="gpt-4",
            temperature=0.5,
        )
        await client.generate([{"role": "user", "content": "Bye"}])
        
        assert client.get_call_count() == 2
        
        calls = client.get_calls()
        assert calls[0]["model"] == "gpt-4"
        assert calls[1]["messages"][0]["content"] == "Bye"
        
        client.reset()
        assert client.get_call_count() == 0
    
    @pytest.mark.asyncio
    async def test_streaming(self):
        """Test streaming responses."""
        client = MockLLMClient(responses=[
            MockLLMResponse(content="Hello, world!")
        ])
        
        chunks = []
        async for chunk in client.stream([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
        
        # Should have content chunks plus final chunk
        assert len(chunks) >= 2
        assert chunks[-1].finish_reason == "stop"
        
        # Reconstruct content
        content = "".join(c.delta for c in chunks)
        assert content == "Hello, world!"


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_test_context(self):
        """Test create_test_context helper."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "result",
        )
        
        ctx = create_test_context(
            "Hello, agent!",
            tools=[tool],
            metadata={"user": "test"},
            params={"mode": "test"},
        )
        
        assert ctx.input_messages[0]["content"] == "Hello, agent!"
        assert ctx.tool_registry.get("test_tool") is not None
        assert ctx.metadata["user"] == "test"
        assert ctx.params["mode"] == "test"
    
    @pytest.mark.asyncio
    async def test_run_agent_test(self):
        """Test run_agent_test helper."""
        
        class TestAgent(AgentRuntime):
            @property
            def key(self) -> str:
                return "test-agent"
            
            async def run(self, ctx: RunContext) -> RunResult:
                await ctx.emit(EventType.RUN_STARTED, {})
                return RunResult(
                    final_output={"echo": ctx.input_messages[0]["content"]}
                )
        
        agent = TestAgent()
        result, ctx = await run_agent_test(agent, "Hello!")
        
        assert result.final_output["echo"] == "Hello!"
        assert len(ctx.get_events()) == 1
        assert ctx.get_events()[0][0] == "run.started"
