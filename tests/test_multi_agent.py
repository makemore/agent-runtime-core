"""
Tests for multi-agent support (agent-as-tool pattern).
"""

import pytest
from uuid import uuid4

from agent_runtime_core import (
    AgentRuntime,
    RunContext,
    RunResult,
    ToolRegistry,
    Tool,
    Message,
)
from agent_runtime_core.multi_agent import (
    AgentTool,
    AgentInvocationResult,
    InvocationMode,
    ContextMode,
    SubAgentContext,
    invoke_agent,
    create_agent_tool_handler,
    register_agent_tools,
    build_sub_agent_messages,
)
from agent_runtime_core.contexts import InMemoryRunContext


class MockSubAgent(AgentRuntime):
    """A simple mock agent for testing."""
    
    def __init__(self, agent_key: str = "mock-sub-agent", response: str = "Sub-agent response"):
        self._key = agent_key
        self._response = response
        self.last_ctx: RunContext = None
        self.call_count = 0
    
    @property
    def key(self) -> str:
        return self._key
    
    async def run(self, ctx: RunContext) -> RunResult:
        self.last_ctx = ctx
        self.call_count += 1
        return RunResult(
            final_output={"response": self._response},
            final_messages=[
                {"role": "assistant", "content": self._response}
            ],
        )


class TestAgentTool:
    """Tests for AgentTool dataclass."""
    
    def test_create_agent_tool(self):
        """Test creating an AgentTool with defaults."""
        agent = MockSubAgent()
        tool = AgentTool(
            agent=agent,
            name="test_agent",
            description="A test agent",
        )
        
        assert tool.name == "test_agent"
        assert tool.description == "A test agent"
        assert tool.invocation_mode == InvocationMode.DELEGATE
        assert tool.context_mode == ContextMode.FULL
        assert tool.max_turns is None
    
    def test_create_agent_tool_with_options(self):
        """Test creating an AgentTool with custom options."""
        agent = MockSubAgent()
        tool = AgentTool(
            agent=agent,
            name="billing_specialist",
            description="Handles billing",
            invocation_mode=InvocationMode.HANDOFF,
            context_mode=ContextMode.MESSAGE_ONLY,
            max_turns=5,
        )
        
        assert tool.invocation_mode == InvocationMode.HANDOFF
        assert tool.context_mode == ContextMode.MESSAGE_ONLY
        assert tool.max_turns == 5
    
    def test_to_tool_schema(self):
        """Test generating OpenAI tool schema."""
        agent = MockSubAgent()
        tool = AgentTool(
            agent=agent,
            name="support_agent",
            description="Handles support questions",
        )
        
        schema = tool.to_tool_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "support_agent"
        assert schema["function"]["description"] == "Handles support questions"
        assert "message" in schema["function"]["parameters"]["properties"]
        assert "context" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["message"]
    
    def test_custom_input_schema(self):
        """Test using a custom input schema."""
        agent = MockSubAgent()
        custom_schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "action": {"type": "string", "enum": ["refund", "cancel"]},
            },
            "required": ["order_id", "action"],
        }
        
        tool = AgentTool(
            agent=agent,
            name="order_agent",
            description="Handles orders",
            input_schema=custom_schema,
        )
        
        schema = tool.to_tool_schema()
        assert schema["function"]["parameters"] == custom_schema


class TestBuildSubAgentMessages:
    """Tests for build_sub_agent_messages function."""
    
    def test_full_context_mode(self):
        """Test FULL context mode includes all history."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "I need help"},
        ]
        
        messages = build_sub_agent_messages(
            context_mode=ContextMode.FULL,
            message="Process my refund",
            conversation_history=history,
        )
        
        # Should include all history + new message
        assert len(messages) == 4
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["content"] == "I need help"
        assert messages[3]["content"] == "Process my refund"
        assert messages[3]["role"] == "user"
    
    def test_full_context_with_additional_context(self):
        """Test FULL mode with additional context."""
        history = [{"role": "user", "content": "Hello"}]
        
        messages = build_sub_agent_messages(
            context_mode=ContextMode.FULL,
            message="Help me",
            conversation_history=history,
            additional_context="Customer is VIP",
        )
        
        assert len(messages) == 2
        assert "Customer is VIP" in messages[1]["content"]
        assert "Help me" in messages[1]["content"]
    
    def test_message_only_mode(self):
        """Test MESSAGE_ONLY mode excludes history."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        
        messages = build_sub_agent_messages(
            context_mode=ContextMode.MESSAGE_ONLY,
            message="Just this message",
            conversation_history=history,
        )
        
        assert len(messages) == 1
        assert messages[0]["content"] == "Just this message"
        assert messages[0]["role"] == "user"
    
    def test_summary_mode_recent_messages(self):
        """Test SUMMARY mode includes recent messages."""
        # Create a long history
        history = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        
        messages = build_sub_agent_messages(
            context_mode=ContextMode.SUMMARY,
            message="New message",
            conversation_history=history,
        )
        
        # Should include last 5 + new message
        assert len(messages) == 6
        assert messages[-1]["content"] == "New message"


class TestSubAgentContext:
    """Tests for SubAgentContext."""
    
    def test_creates_new_run_id(self):
        """Test that sub-agent gets a new run ID."""
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        
        sub_ctx = SubAgentContext(
            parent_ctx=parent_ctx,
            input_messages=[{"role": "user", "content": "test"}],
        )
        
        assert sub_ctx.run_id != parent_ctx.run_id
    
    def test_shares_conversation_id(self):
        """Test that sub-agent shares conversation ID."""
        conv_id = uuid4()
        parent_ctx = InMemoryRunContext(
            run_id=uuid4(),
            conversation_id=conv_id,
        )
        
        sub_ctx = SubAgentContext(
            parent_ctx=parent_ctx,
            input_messages=[],
        )
        
        assert sub_ctx.conversation_id == conv_id
    
    def test_metadata_includes_parent_info(self):
        """Test that metadata includes parent run info."""
        parent_ctx = InMemoryRunContext(
            run_id=uuid4(),
            metadata={"channel": "web"},
        )
        
        sub_ctx = SubAgentContext(
            parent_ctx=parent_ctx,
            input_messages=[],
        )
        
        assert sub_ctx.metadata["is_sub_agent"] is True
        assert sub_ctx.metadata["parent_run_id"] == str(parent_ctx.run_id)
        assert sub_ctx.metadata["channel"] == "web"
    
    def test_cancelled_delegates_to_parent(self):
        """Test that cancelled() checks parent."""
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        sub_ctx = SubAgentContext(parent_ctx=parent_ctx, input_messages=[])
        
        assert sub_ctx.cancelled() is False
        
        parent_ctx.cancel()
        assert sub_ctx.cancelled() is True


@pytest.mark.asyncio
class TestInvokeAgent:
    """Tests for invoke_agent function."""
    
    async def test_delegate_mode_returns_result(self):
        """Test DELEGATE mode returns result to parent."""
        sub_agent = MockSubAgent(response="I processed your request")
        agent_tool = AgentTool(
            agent=sub_agent,
            name="processor",
            description="Processes requests",
            invocation_mode=InvocationMode.DELEGATE,
        )
        
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        
        result = await invoke_agent(
            agent_tool=agent_tool,
            message="Process this",
            parent_ctx=parent_ctx,
        )
        
        assert result.response == "I processed your request"
        assert result.handoff is False
        assert result.sub_agent_key == "mock-sub-agent"
        assert sub_agent.call_count == 1
    
    async def test_handoff_mode_signals_handoff(self):
        """Test HANDOFF mode signals control transfer."""
        sub_agent = MockSubAgent(response="Taking over")
        agent_tool = AgentTool(
            agent=sub_agent,
            name="specialist",
            description="Specialist agent",
            invocation_mode=InvocationMode.HANDOFF,
        )
        
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        
        result = await invoke_agent(
            agent_tool=agent_tool,
            message="Transfer to specialist",
            parent_ctx=parent_ctx,
        )
        
        assert result.handoff is True
        assert result.response == "Taking over"
    
    async def test_full_context_passed_to_sub_agent(self):
        """Test that FULL context mode passes history."""
        sub_agent = MockSubAgent()
        agent_tool = AgentTool(
            agent=sub_agent,
            name="helper",
            description="Helper",
            context_mode=ContextMode.FULL,
        )
        
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]
        
        await invoke_agent(
            agent_tool=agent_tool,
            message="New request",
            parent_ctx=parent_ctx,
            conversation_history=history,
        )
        
        # Sub-agent should have received full history + new message
        sub_messages = sub_agent.last_ctx.input_messages
        assert len(sub_messages) == 3
        assert sub_messages[0]["content"] == "First message"
        assert sub_messages[2]["content"] == "New request"
    
    async def test_emits_tool_call_and_result_events(self):
        """Test that events are emitted for sub-agent invocation."""
        sub_agent = MockSubAgent()
        agent_tool = AgentTool(
            agent=sub_agent,
            name="evented_agent",
            description="Agent with events",
        )
        
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        
        await invoke_agent(
            agent_tool=agent_tool,
            message="Test",
            parent_ctx=parent_ctx,
        )
        
        events = parent_ctx.events
        event_types = [e["event_type"] for e in events]
        
        assert "tool.call" in event_types
        assert "tool.result" in event_types
        
        # Check tool.call event has agent info
        tool_call_event = next(e for e in events if e["event_type"] == "tool.call")
        assert tool_call_event["payload"]["is_agent_tool"] is True
        assert tool_call_event["payload"]["sub_agent_key"] == "mock-sub-agent"


@pytest.mark.asyncio
class TestRegisterAgentTools:
    """Tests for register_agent_tools function."""
    
    async def test_registers_agent_as_tool(self):
        """Test that agent tools are registered in registry."""
        sub_agent = MockSubAgent()
        agent_tool = AgentTool(
            agent=sub_agent,
            name="registered_agent",
            description="A registered agent tool",
        )
        
        registry = ToolRegistry()
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        messages = []
        
        register_agent_tools(
            registry=registry,
            agent_tools=[agent_tool],
            get_conversation_history=lambda: messages,
            parent_ctx=parent_ctx,
        )
        
        # Should be registered
        tool = registry.get("registered_agent")
        assert tool is not None
        assert tool.metadata["is_agent_tool"] is True
        assert tool.metadata["sub_agent_key"] == "mock-sub-agent"
    
    async def test_registered_tool_can_be_executed(self):
        """Test that registered agent tool can be executed."""
        sub_agent = MockSubAgent(response="Executed!")
        agent_tool = AgentTool(
            agent=sub_agent,
            name="executable_agent",
            description="Can be executed",
        )
        
        registry = ToolRegistry()
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        
        register_agent_tools(
            registry=registry,
            agent_tools=[agent_tool],
            get_conversation_history=lambda: [],
            parent_ctx=parent_ctx,
        )
        
        # Execute through registry
        result = await registry.execute("executable_agent", {"message": "Do something"})
        
        assert result["response"] == "Executed!"
        assert sub_agent.call_count == 1
    
    async def test_handoff_result_includes_flag(self):
        """Test that handoff tools return handoff flag."""
        sub_agent = MockSubAgent()
        agent_tool = AgentTool(
            agent=sub_agent,
            name="handoff_agent",
            description="Hands off",
            invocation_mode=InvocationMode.HANDOFF,
        )
        
        registry = ToolRegistry()
        parent_ctx = InMemoryRunContext(run_id=uuid4())
        
        register_agent_tools(
            registry=registry,
            agent_tools=[agent_tool],
            get_conversation_history=lambda: [],
            parent_ctx=parent_ctx,
        )
        
        result = await registry.execute("handoff_agent", {"message": "Transfer"})
        
        assert result["handoff"] is True


class TestInvocationModeEnum:
    """Tests for InvocationMode enum."""
    
    def test_delegate_value(self):
        assert InvocationMode.DELEGATE.value == "delegate"
    
    def test_handoff_value(self):
        assert InvocationMode.HANDOFF.value == "handoff"


class TestContextModeEnum:
    """Tests for ContextMode enum."""
    
    def test_full_value(self):
        assert ContextMode.FULL.value == "full"
    
    def test_summary_value(self):
        assert ContextMode.SUMMARY.value == "summary"
    
    def test_message_only_value(self):
        assert ContextMode.MESSAGE_ONLY.value == "message_only"

