"""
Agent runner - executes agent runs with full lifecycle management.

The runner handles:
- Claiming runs from the queue
- Executing agent runtimes
- Emitting events
- Managing state and checkpoints
- Handling errors and retries
- Cancellation
"""

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from agent_runtime_core.config import get_config
from agent_runtime_core.events.base import EventBus
from agent_runtime_core.interfaces import (
    AgentRuntime,
    EventType,
    ErrorInfo,
    Message,
    RunResult,
    ToolRegistry,
)
from agent_runtime_core.queue.base import RunQueue, QueuedRun
from agent_runtime_core.registry import get_runtime
from agent_runtime_core.state.base import StateStore


@dataclass
class RunnerConfig:
    """Configuration for the agent runner."""
    
    worker_id: str = "worker-1"
    run_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    lease_ttl_seconds: int = 60
    max_retries: int = 3
    retry_backoff_base: int = 2
    retry_backoff_max: int = 300


class RunContextImpl:
    """
    Implementation of RunContext provided to agent runtimes.
    
    This is what agent frameworks use to interact with the runtime.
    """
    
    def __init__(
        self,
        run_id: UUID,
        conversation_id: Optional[UUID],
        input_messages: list[Message],
        params: dict,
        metadata: dict,
        tool_registry: ToolRegistry,
        event_bus: EventBus,
        state_store: StateStore,
        queue: RunQueue,
    ):
        self._run_id = run_id
        self._conversation_id = conversation_id
        self._input_messages = input_messages
        self._params = params
        self._metadata = metadata
        self._tool_registry = tool_registry
        self._event_bus = event_bus
        self._state_store = state_store
        self._queue = queue
    
    @property
    def run_id(self) -> UUID:
        return self._run_id
    
    @property
    def conversation_id(self) -> Optional[UUID]:
        return self._conversation_id
    
    @property
    def input_messages(self) -> list[Message]:
        return self._input_messages
    
    @property
    def params(self) -> dict:
        return self._params
    
    @property
    def metadata(self) -> dict:
        return self._metadata
    
    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry
    
    async def emit(self, event_type: EventType | str, payload: dict) -> None:
        """Emit an event to the event bus."""
        event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
        await self._event_bus.publish(self._run_id, event_type_str, payload)
    
    async def checkpoint(self, state: dict) -> None:
        """Save a state checkpoint."""
        await self._state_store.save_checkpoint(self._run_id, state)
        await self.emit(EventType.STATE_CHECKPOINT, {"state": state})
    
    async def get_state(self) -> Optional[dict]:
        """Get the last checkpointed state."""
        return await self._state_store.get_checkpoint(self._run_id)
    
    def cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        # This is synchronous for easy checking in loops
        # We use asyncio to check the queue
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._queue.is_cancelled(self._run_id))
        except RuntimeError:
            # No event loop running, can't check
            return False


class AgentRunner:
    """
    Runs agent executions with full lifecycle management.
    
    The runner:
    1. Claims runs from the queue
    2. Looks up the appropriate runtime
    3. Executes the runtime with a context
    4. Handles errors, retries, and cancellation
    5. Emits events throughout
    """
    
    def __init__(
        self,
        queue: RunQueue,
        event_bus: EventBus,
        state_store: StateStore,
        config: Optional[RunnerConfig] = None,
    ):
        self.queue = queue
        self.event_bus = event_bus
        self.state_store = state_store
        self.config = config or RunnerConfig()
        self._running = False
        self._current_run: Optional[UUID] = None
    
    async def run_once(self) -> bool:
        """
        Claim and execute a single run.
        
        Returns:
            True if a run was executed, False if queue was empty
        """
        # Claim a run
        queued_run = await self.queue.claim(
            worker_id=self.config.worker_id,
            lease_seconds=self.config.lease_ttl_seconds,
        )
        
        if queued_run is None:
            return False
        
        await self._execute_run(queued_run)
        return True
    
    async def run_loop(self, poll_interval: float = 1.0) -> None:
        """
        Run continuously, processing runs from the queue.
        
        Args:
            poll_interval: Seconds to wait between queue polls
        """
        self._running = True
        
        while self._running:
            try:
                executed = await self.run_once()
                if not executed:
                    await asyncio.sleep(poll_interval)
            except Exception as e:
                # Log error but keep running
                print(f"Error in run loop: {e}")
                await asyncio.sleep(poll_interval)
    
    def stop(self) -> None:
        """Stop the run loop."""
        self._running = False
    
    async def _execute_run(self, queued_run: QueuedRun) -> None:
        """Execute a single run."""
        run_id = queued_run.run_id
        self._current_run = run_id
        
        try:
            # Look up the runtime
            runtime = get_runtime(queued_run.agent_key)
            if runtime is None:
                raise ValueError(f"Unknown agent: {queued_run.agent_key}")
            
            # Update status
            await self.state_store.update_run_status(run_id, "running")
            
            # Build context
            ctx = self._build_context(queued_run)
            
            # Emit started event
            await ctx.emit(EventType.RUN_STARTED, {
                "agent_key": queued_run.agent_key,
                "attempt": queued_run.attempt,
            })
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(run_id)
            )
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    runtime.run(ctx),
                    timeout=self.config.run_timeout_seconds,
                )
                
                # Success!
                await self._handle_success(queued_run, ctx, result)
                
            except asyncio.TimeoutError:
                await self._handle_timeout(queued_run, ctx, runtime)
            except asyncio.CancelledError:
                await self._handle_cancellation(queued_run, ctx, runtime)
            except Exception as e:
                await self._handle_error(queued_run, ctx, runtime, e)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
        
        finally:
            self._current_run = None
    
    def _build_context(self, queued_run: QueuedRun) -> RunContextImpl:
        """Build a RunContext for a queued run."""
        input_data = queued_run.input
        
        return RunContextImpl(
            run_id=queued_run.run_id,
            conversation_id=input_data.get("conversation_id"),
            input_messages=input_data.get("messages", []),
            params=input_data.get("params", {}),
            metadata=queued_run.metadata,
            tool_registry=ToolRegistry(),  # TODO: Load from config
            event_bus=self.event_bus,
            state_store=self.state_store,
            queue=self.queue,
        )
    
    async def _heartbeat_loop(self, run_id: UUID) -> None:
        """Send periodic heartbeats to extend the lease."""
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            
            # Extend lease
            extended = await self.queue.extend_lease(
                run_id=run_id,
                worker_id=self.config.worker_id,
                lease_seconds=self.config.lease_ttl_seconds,
            )
            
            if not extended:
                # Lost the lease
                break
            
            # Emit heartbeat event
            await self.event_bus.publish(
                run_id,
                EventType.RUN_HEARTBEAT.value,
                {"timestamp": datetime.now(timezone.utc).isoformat()},
            )
    
    async def _handle_success(
        self,
        queued_run: QueuedRun,
        ctx: RunContextImpl,
        result: RunResult,
    ) -> None:
        """Handle successful run completion."""
        await self.state_store.update_run_status(queued_run.run_id, "succeeded")
        
        await ctx.emit(EventType.RUN_SUCCEEDED, {
            "final_output": result.final_output,
            "usage": result.usage,
        })
        
        await self.queue.release(
            run_id=queued_run.run_id,
            worker_id=self.config.worker_id,
            success=True,
            output=result.final_output,
        )
    
    async def _handle_timeout(
        self,
        queued_run: QueuedRun,
        ctx: RunContextImpl,
        runtime: AgentRuntime,
    ) -> None:
        """Handle run timeout."""
        await self.state_store.update_run_status(queued_run.run_id, "timed_out")
        
        await ctx.emit(EventType.RUN_TIMED_OUT, {
            "timeout_seconds": self.config.run_timeout_seconds,
        })
        
        await self.queue.release(
            run_id=queued_run.run_id,
            worker_id=self.config.worker_id,
            success=False,
            error={"type": "TimeoutError", "message": "Run timed out"},
        )
    
    async def _handle_cancellation(
        self,
        queued_run: QueuedRun,
        ctx: RunContextImpl,
        runtime: AgentRuntime,
    ) -> None:
        """Handle run cancellation."""
        await runtime.cancel(ctx)
        await self.state_store.update_run_status(queued_run.run_id, "cancelled")
        
        await ctx.emit(EventType.RUN_CANCELLED, {})
        
        await self.queue.release(
            run_id=queued_run.run_id,
            worker_id=self.config.worker_id,
            success=False,
            error={"type": "CancelledError", "message": "Run was cancelled"},
        )
    
    async def _handle_error(
        self,
        queued_run: QueuedRun,
        ctx: RunContextImpl,
        runtime: AgentRuntime,
        error: Exception,
    ) -> None:
        """Handle run error."""
        # Get error info from runtime
        error_info = await runtime.on_error(ctx, error)
        if error_info is None:
            error_info = ErrorInfo(
                type=type(error).__name__,
                message=str(error),
                stack=traceback.format_exc(),
                retriable=True,
            )
        
        error_dict = {
            "type": error_info.type,
            "message": error_info.message,
            "stack": error_info.stack,
            "retriable": error_info.retriable,
            "details": error_info.details,
        }
        
        # Check if we should retry
        if error_info.retriable and queued_run.attempt < self.config.max_retries:
            # Calculate backoff
            delay = min(
                self.config.retry_backoff_base ** queued_run.attempt,
                self.config.retry_backoff_max,
            )
            
            requeued = await self.queue.requeue_for_retry(
                run_id=queued_run.run_id,
                worker_id=self.config.worker_id,
                error=error_dict,
                delay_seconds=delay,
            )
            
            if requeued:
                await self.state_store.update_run_status(queued_run.run_id, "pending")
                return
        
        # No retry - mark as failed
        await self.state_store.update_run_status(queued_run.run_id, "failed")
        
        await ctx.emit(EventType.RUN_FAILED, error_dict)
        
        await self.queue.release(
            run_id=queued_run.run_id,
            worker_id=self.config.worker_id,
            success=False,
            error=error_dict,
        )
