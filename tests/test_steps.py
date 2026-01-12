"""Tests for step execution module."""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from agent_runtime_core.steps import (
    Step,
    StepExecutor,
    StepResult,
    StepStatus,
    ExecutionState,
    StepExecutionError,
    StepCancelledError,
)
from agent_runtime_core.interfaces import EventType
from agent_runtime_core.testing import MockRunContext


class TestStep:
    """Tests for Step dataclass."""

    def test_step_defaults(self):
        """Test Step with default values."""
        async def dummy(ctx, state):
            return "result"

        step = Step("test", dummy)

        assert step.name == "test"
        assert step.fn == dummy
        assert step.retries == 0
        assert step.retry_delay == 1.0
        assert step.timeout is None
        assert step.description is None
        assert step.checkpoint is True

    def test_step_with_options(self):
        """Test Step with custom options."""
        async def dummy(ctx, state):
            return "result"

        step = Step(
            name="fetch",
            fn=dummy,
            retries=3,
            retry_delay=2.0,
            timeout=30.0,
            description="Fetch data from API",
            checkpoint=False,
        )

        assert step.retries == 3
        assert step.retry_delay == 2.0
        assert step.timeout == 30.0
        assert step.description == "Fetch data from API"
        assert step.checkpoint is False


class TestExecutionState:
    """Tests for ExecutionState."""

    def test_state_defaults(self):
        """Test ExecutionState with defaults."""
        state = ExecutionState()

        assert state.current_step_index == 0
        assert state.completed_steps == []
        assert state.step_results == {}
        assert state.custom_state == {}

    def test_state_serialization(self):
        """Test state to_dict and from_dict."""
        state = ExecutionState(
            current_step_index=2,
            completed_steps=["step1", "step2"],
            step_results={"step1": "result1", "step2": "result2"},
            custom_state={"key": "value"},
        )

        data = state.to_dict()
        restored = ExecutionState.from_dict(data)

        assert restored.current_step_index == 2
        assert restored.completed_steps == ["step1", "step2"]
        assert restored.step_results == {"step1": "result1", "step2": "result2"}
        assert restored.custom_state == {"key": "value"}


class TestStepExecutor:
    """Tests for StepExecutor."""

    @pytest.mark.asyncio
    async def test_execute_single_step(self):
        """Test executing a single step."""
        ctx = MockRunContext()

        async def my_step(ctx, state):
            return {"data": "hello"}

        executor = StepExecutor(ctx)
        results = await executor.run([Step("my_step", my_step)])

        assert results == {"my_step": {"data": "hello"}}

    @pytest.mark.asyncio
    async def test_execute_multiple_steps(self):
        """Test executing multiple steps in sequence."""
        ctx = MockRunContext()
        call_order = []

        async def step1(ctx, state):
            call_order.append("step1")
            return "result1"

        async def step2(ctx, state):
            call_order.append("step2")
            return "result2"

        async def step3(ctx, state):
            call_order.append("step3")
            return "result3"

        executor = StepExecutor(ctx)
        results = await executor.run([
            Step("step1", step1),
            Step("step2", step2),
            Step("step3", step3),
        ])

        assert call_order == ["step1", "step2", "step3"]
        assert results == {
            "step1": "result1",
            "step2": "result2",
            "step3": "result3",
        }

    @pytest.mark.asyncio
    async def test_step_receives_custom_state(self):
        """Test that steps receive and can modify custom state."""
        ctx = MockRunContext()

        async def step1(ctx, state):
            state["counter"] = 1
            return "done"

        async def step2(ctx, state):
            state["counter"] += 1
            return state["counter"]

        executor = StepExecutor(ctx)
        results = await executor.run(
            [Step("step1", step1), Step("step2", step2)],
            initial_state={"counter": 0},
        )

        assert results["step2"] == 2

    @pytest.mark.asyncio
    async def test_step_emits_events(self):
        """Test that step execution emits proper events."""
        ctx = MockRunContext()

        async def my_step(ctx, state):
            return "done"

        executor = StepExecutor(ctx)
        await executor.run([Step("my_step", my_step, description="Test step")])

        events = ctx.get_events()
        event_types = [event_type for event_type, _ in events]

        assert EventType.STEP_STARTED.value in event_types
        assert EventType.STEP_COMPLETED.value in event_types
        assert EventType.PROGRESS_UPDATE.value in event_types

    @pytest.mark.asyncio
    async def test_step_retry_on_failure(self):
        """Test that steps retry on failure."""
        ctx = MockRunContext()
        attempts = []

        async def flaky_step(ctx, state):
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Temporary failure")
            return "success"

        executor = StepExecutor(ctx)
        results = await executor.run([
            Step("flaky", flaky_step, retries=3, retry_delay=0.01)
        ])

        assert len(attempts) == 3
        assert results["flaky"] == "success"

        # Check retry events
        events = ctx.get_events()
        retry_events = [
            (event_type, payload)
            for event_type, payload in events
            if event_type == EventType.STEP_RETRYING.value
        ]
        assert len(retry_events) == 2  # 2 retries before success

    @pytest.mark.asyncio
    async def test_step_fails_after_max_retries(self):
        """Test that step fails after exhausting retries."""
        ctx = MockRunContext()

        async def always_fails(ctx, state):
            raise ValueError("Always fails")

        executor = StepExecutor(ctx)

        with pytest.raises(StepExecutionError) as exc_info:
            await executor.run([
                Step("failing", always_fails, retries=2, retry_delay=0.01)
            ])

        assert exc_info.value.step_name == "failing"
        assert exc_info.value.attempts == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_step_timeout(self):
        """Test step timeout handling."""
        ctx = MockRunContext()

        async def slow_step(ctx, state):
            await asyncio.sleep(10)
            return "done"

        executor = StepExecutor(ctx)

        with pytest.raises(StepExecutionError) as exc_info:
            await executor.run([
                Step("slow", slow_step, timeout=0.1)
            ])

        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Test cancellation during execution."""
        ctx = MockRunContext()

        async def step1(ctx, state):
            ctx.cancel()  # Cancel during first step
            return "done"

        async def step2(ctx, state):
            return "should not run"

        executor = StepExecutor(ctx)

        with pytest.raises(StepCancelledError):
            await executor.run([
                Step("step1", step1),
                Step("step2", step2),
            ])

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume(self):
        """Test checkpointing and resuming execution."""
        ctx = MockRunContext()
        call_order = []

        async def step1(ctx, state):
            call_order.append("step1")
            return "result1"

        async def step2(ctx, state):
            call_order.append("step2")
            return "result2"

        # First run - complete step1
        executor1 = StepExecutor(ctx)
        await executor1.run([Step("step1", step1)])

        # Second run - should skip step1, run step2
        executor2 = StepExecutor(ctx)
        results = await executor2.run([
            Step("step1", step1),
            Step("step2", step2),
        ])

        # step1 should only be called once (first run)
        # step2 should be called in second run
        assert call_order == ["step1", "step2"]
        assert results["step2"] == "result2"

        # Check skip event
        events = ctx.get_events()
        skip_events = [
            (event_type, payload)
            for event_type, payload in events
            if event_type == EventType.STEP_SKIPPED.value
        ]
        assert len(skip_events) == 1
        _, payload = skip_events[0]
        assert payload["step_name"] == "step1"

    @pytest.mark.asyncio
    async def test_resume_disabled(self):
        """Test that resume can be disabled."""
        ctx = MockRunContext()
        call_count = 0

        async def my_step(ctx, state):
            nonlocal call_count
            call_count += 1
            return "done"

        # First run
        executor1 = StepExecutor(ctx)
        await executor1.run([Step("my_step", my_step)])

        # Second run with resume=False
        executor2 = StepExecutor(ctx)
        await executor2.run([Step("my_step", my_step)], resume=False)

        # Step should be called twice
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_update_custom_state(self):
        """Test updating custom state via executor."""
        ctx = MockRunContext()

        async def my_step(ctx, state):
            return state.get("value", 0)

        executor = StepExecutor(ctx)
        executor._state = ExecutionState(custom_state={"value": 42})
        executor.update_custom_state({"value": 100})

        assert executor.state.custom_state["value"] == 100

    @pytest.mark.asyncio
    async def test_progress_percentage(self):
        """Test progress percentage calculation."""
        ctx = MockRunContext()

        async def dummy(ctx, state):
            return "done"

        executor = StepExecutor(ctx)
        await executor.run([
            Step("step1", dummy),
            Step("step2", dummy),
            Step("step3", dummy),
            Step("step4", dummy),
        ])

        events = ctx.get_events()
        progress_events = [
            (event_type, payload)
            for event_type, payload in events
            if event_type == EventType.PROGRESS_UPDATE.value
        ]

        # Check progress percentages
        percentages = [payload["progress_percent"] for _, payload in progress_events]
        assert percentages == [0.0, 25.0, 50.0, 75.0]
