"""
Tests for concrete RunContext implementations.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from agent_runtime_core.contexts import InMemoryRunContext, FileRunContext
from agent_runtime_core.interfaces import EventType, ToolRegistry
from agent_runtime_core.steps import Step, StepExecutor


class TestInMemoryRunContext:
    """Tests for InMemoryRunContext."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        ctx = InMemoryRunContext()
        
        assert ctx.run_id is not None
        assert ctx.conversation_id is None
        assert ctx.input_messages == []
        assert ctx.params == {}
        assert ctx.metadata == {}
        assert isinstance(ctx.tool_registry, ToolRegistry)
        assert ctx.cancelled() is False
    
    def test_init_with_values(self):
        """Test initialization with custom values."""
        run_id = uuid4()
        conv_id = uuid4()
        messages = [{"role": "user", "content": "Hello"}]
        params = {"temperature": 0.7}
        metadata = {"user_id": "123"}
        
        ctx = InMemoryRunContext(
            run_id=run_id,
            conversation_id=conv_id,
            input_messages=messages,
            params=params,
            metadata=metadata,
        )
        
        assert ctx.run_id == run_id
        assert ctx.conversation_id == conv_id
        assert ctx.input_messages == messages
        assert ctx.params == params
        assert ctx.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_checkpoint_and_get_state(self):
        """Test checkpointing and retrieving state."""
        ctx = InMemoryRunContext()
        
        # Initially no state
        state = await ctx.get_state()
        assert state is None
        
        # Checkpoint some state
        await ctx.checkpoint({"step": 1, "data": "test"})
        
        # Retrieve state
        state = await ctx.get_state()
        assert state == {"step": 1, "data": "test"}
        
        # Update state
        await ctx.checkpoint({"step": 2, "data": "updated"})
        state = await ctx.get_state()
        assert state == {"step": 2, "data": "updated"}
    
    @pytest.mark.asyncio
    async def test_emit_events(self):
        """Test event emission."""
        events_received = []
        
        def on_event(event_type, payload):
            events_received.append((event_type, payload))
        
        ctx = InMemoryRunContext(on_event=on_event)
        
        await ctx.emit(EventType.RUN_STARTED, {"agent": "test"})
        await ctx.emit("custom.event", {"data": "value"})
        
        # Check callback was called
        assert len(events_received) == 2
        assert events_received[0] == ("run.started", {"agent": "test"})
        assert events_received[1] == ("custom.event", {"data": "value"})
        
        # Check events are stored
        assert len(ctx.events) == 2
        assert ctx.events[0]["event_type"] == "run.started"
        assert ctx.events[1]["event_type"] == "custom.event"
    
    def test_cancellation(self):
        """Test cancellation."""
        ctx = InMemoryRunContext()
        
        assert ctx.cancelled() is False
        ctx.cancel()
        assert ctx.cancelled() is True


class TestFileRunContext:
    """Tests for FileRunContext."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_init_creates_directory(self, temp_dir):
        """Test that init creates the checkpoint directory."""
        checkpoint_dir = Path(temp_dir) / "nested" / "checkpoints"
        ctx = FileRunContext(checkpoint_dir=str(checkpoint_dir))
        
        assert checkpoint_dir.exists()
    
    @pytest.mark.asyncio
    async def test_checkpoint_and_get_state(self, temp_dir):
        """Test checkpointing and retrieving state from file."""
        run_id = uuid4()
        ctx = FileRunContext(run_id=run_id, checkpoint_dir=temp_dir)
        
        # Initially no state
        state = await ctx.get_state()
        assert state is None
        
        # Checkpoint some state
        await ctx.checkpoint({"step": 1, "data": "test"})
        
        # Verify file was created
        checkpoint_path = Path(temp_dir) / f"{run_id}.json"
        assert checkpoint_path.exists()
        
        # Retrieve state
        state = await ctx.get_state()
        assert state == {"step": 1, "data": "test"}
        
        # Create new context with same run_id - should load existing state
        ctx2 = FileRunContext(run_id=run_id, checkpoint_dir=temp_dir)
        state2 = await ctx2.get_state()
        assert state2 == {"step": 1, "data": "test"}
    
    @pytest.mark.asyncio
    async def test_emit_events_to_file(self, temp_dir):
        """Test event emission to file."""
        run_id = uuid4()
        ctx = FileRunContext(run_id=run_id, checkpoint_dir=temp_dir)
        
        await ctx.emit(EventType.RUN_STARTED, {"agent": "test"})
        await ctx.emit(EventType.STEP_COMPLETED, {"step": "fetch"})
        
        # Verify events file was created
        events_path = Path(temp_dir) / f"{run_id}_events.jsonl"
        assert events_path.exists()
        
        # Read events back
        events = ctx.get_events()
        assert len(events) == 2
        assert events[0]["event_type"] == "run.started"
        assert events[1]["event_type"] == "step.completed"
    
    @pytest.mark.asyncio
    async def test_clear(self, temp_dir):
        """Test clearing checkpoint and events."""
        run_id = uuid4()
        ctx = FileRunContext(run_id=run_id, checkpoint_dir=temp_dir)
        
        await ctx.checkpoint({"data": "test"})
        await ctx.emit(EventType.RUN_STARTED, {})
        
        checkpoint_path = Path(temp_dir) / f"{run_id}.json"
        events_path = Path(temp_dir) / f"{run_id}_events.jsonl"
        
        assert checkpoint_path.exists()
        assert events_path.exists()
        
        ctx.clear()
        
        assert not checkpoint_path.exists()
        assert not events_path.exists()
        assert await ctx.get_state() is None


class TestContextsWithStepExecutor:
    """Test that contexts work correctly with StepExecutor."""
    
    @pytest.mark.asyncio
    async def test_in_memory_context_with_executor(self):
        """Test InMemoryRunContext with StepExecutor."""
        ctx = InMemoryRunContext()
        
        async def step1(ctx, state):
            state["step1"] = "done"
            return "step1_result"
        
        async def step2(ctx, state):
            assert state["step1"] == "done"
            return "step2_result"
        
        executor = StepExecutor(ctx)
        results = await executor.run([
            Step("step1", step1),
            Step("step2", step2),
        ])
        
        assert results["step1"] == "step1_result"
        assert results["step2"] == "step2_result"
    
    @pytest.mark.asyncio
    async def test_file_context_resume(self):
        """Test FileRunContext resume capability."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = uuid4()
            execution_count = {"step1": 0, "step2": 0}
            
            async def step1(ctx, state):
                execution_count["step1"] += 1
                return "step1_result"
            
            async def step2(ctx, state):
                execution_count["step2"] += 1
                # Simulate failure on first attempt
                if execution_count["step2"] == 1:
                    raise RuntimeError("Simulated failure")
                return "step2_result"
            
            # First run - step1 succeeds, step2 fails
            ctx1 = FileRunContext(run_id=run_id, checkpoint_dir=temp_dir)
            executor1 = StepExecutor(ctx1)
            
            with pytest.raises(Exception):
                await executor1.run([
                    Step("step1", step1),
                    Step("step2", step2),
                ])
            
            assert execution_count["step1"] == 1
            assert execution_count["step2"] == 1
            
            # Second run - should resume from checkpoint, skip step1
            ctx2 = FileRunContext(run_id=run_id, checkpoint_dir=temp_dir)
            executor2 = StepExecutor(ctx2)
            
            results = await executor2.run([
                Step("step1", step1),
                Step("step2", step2),
            ], resume=True)
            
            # step1 should NOT have been re-executed
            assert execution_count["step1"] == 1
            # step2 should have been retried
            assert execution_count["step2"] == 2
            
            assert results["step1"] == "step1_result"
            assert results["step2"] == "step2_result"

