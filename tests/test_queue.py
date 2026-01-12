"""Test queue implementations."""

import pytest
from uuid import uuid4

from agent_runtime_core.queue import InMemoryQueue


@pytest.fixture
def queue():
    """Create a fresh queue for each test."""
    q = InMemoryQueue(max_retries=3)
    yield q
    q.clear()


@pytest.mark.asyncio
async def test_enqueue_and_claim(queue):
    """Test enqueueing and claiming a run."""
    run_id = uuid4()
    
    await queue.enqueue(
        run_id=run_id,
        agent_key="test-agent",
        input={"messages": [{"role": "user", "content": "Hello"}]},
    )
    
    claimed = await queue.claim(worker_id="worker-1", lease_seconds=60)
    
    assert claimed is not None
    assert claimed.run_id == run_id
    assert claimed.agent_key == "test-agent"


@pytest.mark.asyncio
async def test_claim_returns_none_when_empty(queue):
    """Test that claim returns None when queue is empty."""
    claimed = await queue.claim(worker_id="worker-1", lease_seconds=60)
    assert claimed is None


@pytest.mark.asyncio
async def test_release_success(queue):
    """Test releasing a run successfully."""
    run_id = uuid4()
    
    await queue.enqueue(run_id=run_id, agent_key="test-agent", input={})
    await queue.claim(worker_id="worker-1", lease_seconds=60)
    
    await queue.release(
        run_id=run_id,
        worker_id="worker-1",
        success=True,
        output={"result": "done"},
    )
    
    # Should not be claimable again
    claimed = await queue.claim(worker_id="worker-2", lease_seconds=60)
    assert claimed is None


@pytest.mark.asyncio
async def test_cancel(queue):
    """Test cancelling a run."""
    run_id = uuid4()
    
    await queue.enqueue(run_id=run_id, agent_key="test-agent", input={})
    
    result = await queue.cancel(run_id)
    assert result is True
    
    is_cancelled = await queue.is_cancelled(run_id)
    assert is_cancelled is True


@pytest.mark.asyncio
async def test_requeue_for_retry(queue):
    """Test requeuing a run for retry."""
    run_id = uuid4()
    
    await queue.enqueue(run_id=run_id, agent_key="test-agent", input={})
    await queue.claim(worker_id="worker-1", lease_seconds=60)
    
    requeued = await queue.requeue_for_retry(
        run_id=run_id,
        worker_id="worker-1",
        error={"type": "TestError", "message": "Test"},
    )
    
    assert requeued is True
    
    # Should be claimable again
    claimed = await queue.claim(worker_id="worker-2", lease_seconds=60)
    assert claimed is not None
    assert claimed.attempt == 2
