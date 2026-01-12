"""
Step executor for long-running multi-step agent operations.

This module provides a structured way to execute multi-step operations
with automatic checkpointing, resume capability, retries, and progress
reporting.

Example usage:
    from agent_runtime_core.steps import StepExecutor, Step

    class MyAgent(AgentRuntime):
        async def run(self, ctx: RunContext) -> RunResult:
            executor = StepExecutor(ctx)

            result = await executor.run([
                Step("fetch", self.fetch_data),
                Step("process", self.process_data, retries=3),
                Step("validate", self.validate),
            ])

            return RunResult(final_output=result)
"""

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union
from uuid import UUID, uuid4

from agent_runtime_core.interfaces import EventType, RunContext


class StepStatus(str, Enum):
    """Status of a step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


# Type for step functions: async def step_fn(ctx, state) -> result
StepFunction = Callable[[RunContext, dict], Awaitable[Any]]


@dataclass
class Step:
    """
    Definition of a single step in a multi-step operation.

    Args:
        name: Unique identifier for this step
        fn: Async function to execute. Receives (ctx, state) and returns result.
        retries: Number of retry attempts on failure (default: 0)
        retry_delay: Seconds to wait between retries (default: 1.0)
        timeout: Optional timeout in seconds for this step
        description: Human-readable description for progress reporting
        checkpoint: Whether to checkpoint after this step (default: True)
    """

    name: str
    fn: StepFunction
    retries: int = 0
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    description: Optional[str] = None
    checkpoint: bool = True


@dataclass
class StepResult:
    """Result of executing a single step."""

    name: str
    status: StepStatus
    result: Any = None
    error: Optional[str] = None
    attempts: int = 1
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


@dataclass
class ExecutionState:
    """
    State of a multi-step execution.

    This is what gets checkpointed and can be used to resume.
    """

    execution_id: UUID = field(default_factory=uuid4)
    current_step_index: int = 0
    completed_steps: list[str] = field(default_factory=list)
    step_results: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    custom_state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for checkpointing."""
        return {
            "execution_id": str(self.execution_id),
            "current_step_index": self.current_step_index,
            "completed_steps": self.completed_steps,
            "step_results": self.step_results,
            "started_at": self.started_at.isoformat(),
            "custom_state": self.custom_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionState":
        """Restore from checkpointed dictionary."""
        return cls(
            execution_id=UUID(data["execution_id"]),
            current_step_index=data["current_step_index"],
            completed_steps=data["completed_steps"],
            step_results=data["step_results"],
            started_at=datetime.fromisoformat(data["started_at"]),
            custom_state=data.get("custom_state", {}),
        )


class StepExecutionError(Exception):
    """Raised when step execution fails after all retries."""

    def __init__(self, step_name: str, message: str, attempts: int):
        self.step_name = step_name
        self.attempts = attempts
        super().__init__(f"Step '{step_name}' failed after {attempts} attempts: {message}")


class StepCancelledError(Exception):
    """Raised when execution is cancelled."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        super().__init__(f"Execution cancelled during step '{step_name}'")


class StepExecutor:
    """
    Executes a sequence of steps with checkpointing and resume capability.

    Features:
    - Automatic checkpointing after each step
    - Resume from last checkpoint on restart
    - Per-step retries with configurable delay
    - Progress reporting via events
    - Cancellation support
    - Step-level timeouts

    Example:
        executor = StepExecutor(ctx)

        result = await executor.run([
            Step("fetch", fetch_data),
            Step("process", process_data, retries=3),
            Step("save", save_results),
        ])
    """

    def __init__(
        self,
        ctx: RunContext,
        *,
        checkpoint_key: str = "_step_executor_state",
        cancel_check_interval: float = 0.5,
    ):
        """
        Initialize the step executor.

        Args:
            ctx: The run context from the agent runtime
            checkpoint_key: Key used for storing execution state
            cancel_check_interval: How often to check for cancellation (seconds)
        """
        self.ctx = ctx
        self.checkpoint_key = checkpoint_key
        self.cancel_check_interval = cancel_check_interval
        self._state: Optional[ExecutionState] = None

    async def run(
        self,
        steps: list[Step],
        *,
        initial_state: Optional[dict] = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a sequence of steps.

        Args:
            steps: List of steps to execute
            initial_state: Optional initial custom state
            resume: Whether to resume from checkpoint if available

        Returns:
            Dictionary mapping step names to their results

        Raises:
            StepExecutionError: If a step fails after all retries
            StepCancelledError: If execution is cancelled
        """
        # Try to resume from checkpoint
        if resume:
            self._state = await self._load_state()

        if self._state is None:
            self._state = ExecutionState(
                custom_state=initial_state or {}
            )

        total_steps = len(steps)

        for i, step in enumerate(steps):
            # Skip already completed steps
            if step.name in self._state.completed_steps:
                await self.ctx.emit(EventType.STEP_SKIPPED, {
                    "step_name": step.name,
                    "step_index": i,
                    "total_steps": total_steps,
                    "reason": "already_completed",
                })
                continue

            # Check for cancellation
            if self.ctx.cancelled():
                raise StepCancelledError(step.name)

            # Update state
            self._state.current_step_index = i

            # Execute the step
            result = await self._execute_step(step, i, total_steps)

            # Record completion
            self._state.completed_steps.append(step.name)
            self._state.step_results[step.name] = result.result

            # Checkpoint if enabled
            if step.checkpoint:
                await self._save_state()

        return self._state.step_results

    async def _execute_step(
        self,
        step: Step,
        index: int,
        total: int,
    ) -> StepResult:
        """Execute a single step with retries."""
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= step.retries:
            attempts += 1

            # Emit started event
            await self.ctx.emit(EventType.STEP_STARTED, {
                "step_name": step.name,
                "step_index": index,
                "total_steps": total,
                "attempt": attempts,
                "max_attempts": step.retries + 1,
                "description": step.description,
            })

            # Emit progress
            await self.ctx.emit(EventType.PROGRESS_UPDATE, {
                "step_name": step.name,
                "step_index": index,
                "total_steps": total,
                "progress_percent": (index / total) * 100,
                "description": step.description or f"Executing {step.name}",
            })

            started_at = datetime.utcnow()

            try:
                # Execute with optional timeout
                if step.timeout:
                    result = await asyncio.wait_for(
                        step.fn(self.ctx, self._state.custom_state),
                        timeout=step.timeout,
                    )
                else:
                    result = await step.fn(self.ctx, self._state.custom_state)

                completed_at = datetime.utcnow()
                duration_ms = (completed_at - started_at).total_seconds() * 1000

                # Emit completed event
                await self.ctx.emit(EventType.STEP_COMPLETED, {
                    "step_name": step.name,
                    "step_index": index,
                    "total_steps": total,
                    "attempt": attempts,
                    "duration_ms": duration_ms,
                })

                return StepResult(
                    name=step.name,
                    status=StepStatus.COMPLETED,
                    result=result,
                    attempts=attempts,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                )

            except asyncio.CancelledError:
                raise StepCancelledError(step.name)

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step.timeout}s"

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"

            # Check if we should retry
            if attempts <= step.retries:
                await self.ctx.emit(EventType.STEP_RETRYING, {
                    "step_name": step.name,
                    "step_index": index,
                    "attempt": attempts,
                    "max_attempts": step.retries + 1,
                    "error": last_error,
                    "retry_delay": step.retry_delay,
                })
                await asyncio.sleep(step.retry_delay)

        # All retries exhausted
        await self.ctx.emit(EventType.STEP_FAILED, {
            "step_name": step.name,
            "step_index": index,
            "total_steps": total,
            "attempts": attempts,
            "error": last_error,
        })

        raise StepExecutionError(step.name, last_error or "Unknown error", attempts)

    async def _load_state(self) -> Optional[ExecutionState]:
        """Load execution state from checkpoint."""
        checkpoint = await self.ctx.get_state()
        if checkpoint and self.checkpoint_key in checkpoint:
            try:
                return ExecutionState.from_dict(checkpoint[self.checkpoint_key])
            except (KeyError, ValueError):
                return None
        return None

    async def _save_state(self) -> None:
        """Save execution state to checkpoint."""
        checkpoint = await self.ctx.get_state() or {}
        checkpoint[self.checkpoint_key] = self._state.to_dict()
        await self.ctx.checkpoint(checkpoint)

    @property
    def state(self) -> Optional[ExecutionState]:
        """Get the current execution state."""
        return self._state

    def update_custom_state(self, updates: dict) -> None:
        """Update custom state (will be checkpointed with next step)."""
        if self._state:
            self._state.custom_state.update(updates)

